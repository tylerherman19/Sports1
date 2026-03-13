"""
export_json.py
Aggregates all computed values and writes JSON files to docs/data/.
All files are consumed by the browser-side JavaScript.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Any

log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "data")


def _write(filename: str, data: Any) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Wrote {path}")


def export_games(
    espn_games: List[dict],
    team_elos: Dict[str, dict],          # team -> {elo_prob, elo_diff, ...}
    bayesian_posteriors: Dict[str, dict],
    pythagorean_data: Dict[str, dict],
    efficiency_data: Dict[str, dict],
    adjustments_data: Dict[str, dict],   # game_id -> adjustment dict
    lr_weights: dict,
    xgb_probs: Dict[str, float],
    odds_data: List[dict],
    standings: Dict[str, dict],
) -> None:
    """Write docs/data/games.json"""

    # Build odds lookup by home team
    odds_lookup = {}
    for o in odds_data:
        # Match by team name abbreviation (best effort)
        key = o.get("home_team_full", "")
        odds_lookup[key] = o

    from data_fetcher import american_to_implied
    from ensemble import elo_to_prob, compute_edge, kelly_fraction

    games_out = []
    for game in espn_games:
        home = game["home_team"]
        away = game["away_team"]
        game_key = f"{home}_vs_{away}"

        home_elo_data = team_elos.get(home, {})
        away_elo_data = team_elos.get(away, {})

        home_mu = home_elo_data.get("mu", 1500.0)
        away_mu = away_elo_data.get("mu", 1500.0)
        home_sigma = home_elo_data.get("sigma", 75.0)
        away_sigma = away_elo_data.get("sigma", 75.0)

        elo_diff = home_elo_data.get("elo_diff", home_mu - away_mu)
        elo_prob = elo_to_prob(elo_diff)

        home_pyth = pythagorean_data.get(home, {})
        away_pyth = pythagorean_data.get(away, {})
        pyth_home = home_pyth.get("expectation", 0.5)
        pyth_away = away_pyth.get("expectation", 0.5)
        pyth_total = pyth_home + pyth_away
        pyth_prob = pyth_home / pyth_total if pyth_total > 0 else 0.5

        home_eff = efficiency_data.get(home, {})
        away_eff = efficiency_data.get(away, {})
        eff_home = home_eff.get("elo_equiv", 1500.0)
        eff_away = away_eff.get("elo_equiv", 1500.0)
        eff_prob = elo_to_prob(eff_home - eff_away)

        adjs = adjustments_data.get(game_key, {})

        # Build features dict for LR
        home_stand = standings.get(home, {})
        away_stand = standings.get(away, {})
        gp_h = home_stand.get("games_played", 9)
        gp_a = away_stand.get("games_played", 9)
        last5_h = home_elo_data.get("last5_win_rate", 0.5)
        last5_a = away_elo_data.get("last5_win_rate", 0.5)

        features = {
            "elo_diff": round(elo_diff, 2),
            "offensive_rating_diff": round(
                home_eff.get("off_efficiency", 1.0) - away_eff.get("off_efficiency", 1.0), 4
            ),
            "defensive_rating_diff": round(
                home_eff.get("def_efficiency", 1.0) - away_eff.get("def_efficiency", 1.0), 4
            ),
            "home_field": 1.0,
            "rest_days_diff": adjs.get("home_rest_days", 7) - adjs.get("away_rest_days", 7),
            "pythagorean_diff": round(pyth_home - pyth_away, 4),
            "net_efficiency_diff": round(
                home_eff.get("net_efficiency", 0) - away_eff.get("net_efficiency", 0), 4
            ),
            "turnover_diff_adjusted": round(
                adjs.get("home_to_adj", 0) - adjs.get("away_to_adj", 0), 2
            ),
            "travel_distance_diff": round(adjs.get("away_travel_dist", 0), 1),
            "last5_win_rate_diff": round(last5_h - last5_a, 4),
        }

        # Implied LR prob from stored weights
        lr_prob = _apply_lr_weights(features, lr_weights)
        xgb_prob = xgb_probs.get(game_key, lr_prob)  # fallback to LR

        # Market data
        market = _find_market_odds(odds_data, home, away, game)

        # Edge
        model_prob_ensemble = (lr_prob * 0.30 + xgb_prob * 0.25 + elo_prob * 0.20 + pyth_prob * 0.15 + eff_prob * 0.10)
        market_implied = market.get("home_implied", model_prob_ensemble)
        edge = compute_edge(model_prob_ensemble, market_implied)

        kelly = kelly_fraction(
            model_prob_ensemble,
            market.get("home_american", -110),
        ) if market.get("home_american") else None

        games_out.append({
            "game_id": game["game_id"],
            "home_team": home,
            "away_team": away,
            "home_team_name": game.get("home_team_name", home),
            "away_team_name": game.get("away_team_name", away),
            "home_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{home.lower()}.png",
            "away_logo": f"https://a.espncdn.com/i/teamlogos/nfl/500/{away.lower()}.png",
            "kickoff": game.get("kickoff", ""),
            "status": game.get("status", ""),
            "venue": game.get("venue", ""),
            "neutral_site": game.get("neutral_site", False),
            "home_elo": round(home_mu, 1),
            "away_elo": round(away_mu, 1),
            "home_sigma": round(home_sigma, 1),
            "away_sigma": round(away_sigma, 1),
            "features": features,
            "adjustments": adjs,
            "base_probs": {
                "lr_prob": round(lr_prob, 4),
                "xgb_prob": round(xgb_prob, 4),
                "elo_prob": round(elo_prob, 4),
                "pyth_prob": round(pyth_prob, 4),
                "eff_prob": round(eff_prob, 4),
            },
            "ensemble_prob": round(model_prob_ensemble, 4),
            "market": market,
            "edge": round(edge, 4),
            "kelly_pct": round(kelly * 100, 2) if kelly is not None else None,
        })

    _write("games.json", {
        "updated": datetime.now(timezone.utc).isoformat(),
        "games": games_out,
    })


def _apply_lr_weights(features: dict, lr_weights: dict) -> float:
    """Apply logistic regression weights to feature dict to get probability."""
    import math
    coeffs = lr_weights.get("coefficients", {})
    intercept = lr_weights.get("intercept", 0.0)
    linear = intercept + sum(coeffs.get(k, 0.0) * features.get(k, 0.0) for k in coeffs)
    return 1.0 / (1.0 + math.exp(-linear))


def _find_market_odds(odds_data: list, home: str, away: str, game: dict) -> dict:
    """Find best market odds for this matchup."""
    home_name = game.get("home_team_name", home)
    away_name = game.get("away_team_name", away)
    for o in odds_data:
        hf = o.get("home_team_full", "")
        af = o.get("away_team_full", "")
        if (home.lower() in hf.lower() or home_name.lower() in hf.lower()):
            return {
                "home_american": o.get("home_american"),
                "away_american": o.get("away_american"),
                "home_implied": round(o.get("home_implied", 0.5), 4),
                "away_implied": round(o.get("away_implied", 0.5), 4),
            }
    return {}


def export_team_ratings(
    team_elos: Dict[str, float],
    bayesian_posteriors: Dict[str, dict],
    pythagorean_data: Dict[str, dict],
    efficiency_data: Dict[str, dict],
    hierarchical_data: Dict[str, dict],
    standings: Dict[str, dict],
    season_games: List[dict],
    monte_carlo_playoffs: Dict[str, float],
) -> None:
    """Write docs/data/team_ratings.json"""
    teams_out = []
    for team in sorted(team_elos.keys()):
        elo = team_elos[team]
        posterior = bayesian_posteriors.get(team, {"mu": elo, "sigma": 75.0, "games_played": 0})
        pyth = pythagorean_data.get(team, {})
        eff = efficiency_data.get(team, {})
        hier = hierarchical_data.get(team, {})
        stand = standings.get(team, {})
        gp = stand.get("games_played", 0)
        wins = stand.get("wins", 0)
        losses = stand.get("losses", 0)
        ties = stand.get("ties", 0)

        # Last 5 games win rate
        team_games = [
            g for g in season_games
            if (g.get("home_team") == team or g.get("away_team") == team)
            and (g.get("home_score", 0) > 0 or g.get("away_score", 0) > 0)
        ]
        last5 = team_games[-5:]
        last5_wins = sum(
            1 for g in last5
            if (g["home_team"] == team and g["home_score"] > g["away_score"])
            or (g["away_team"] == team and g["away_score"] > g["home_score"])
        )
        last5_win_rate = last5_wins / max(1, len(last5))

        # ELO trend (last 3 weeks)
        elo_trend = "neutral"
        if len(team_games) >= 3:
            from elo_model import DEFAULT_ELO
            recent_wins = sum(
                1 for g in team_games[-3:]
                if (g["home_team"] == team and g["home_score"] > g["away_score"])
                or (g["away_team"] == team and g["away_score"] > g["home_score"])
            )
            elo_trend = "up" if recent_wins >= 2 else ("down" if recent_wins == 0 else "neutral")

        # recent_form_elo: base ELO shifted by recent win-rate signal
        # last5_win_rate 1.0 → +50 ELO, 0.5 → no change, 0.0 → -50 ELO
        base_elo = posterior["mu"]
        recent_form_elo = round(base_elo + (last5_win_rate - 0.5) * 100, 1)

        teams_out.append({
            "abbrev": team,
            "record": f"{wins}-{losses}" + (f"-{ties}" if ties > 0 else ""),
            "games_played": gp,
            "wins": wins,
            "losses": losses,
            "elo": {
                "mu": round(posterior["mu"], 1),
                "sigma": round(posterior["sigma"], 1),
            },
            "pyth": pyth,
            "efficiency": eff,
            "hierarchical": hier,
            "playoff_prob": round(monte_carlo_playoffs.get(team, 0.5), 4),
            "last5_win_rate": round(last5_win_rate, 4),
            "recent_form_elo": recent_form_elo,
            "elo_trend": elo_trend,
        })

    # Sort by ELO descending
    teams_out.sort(key=lambda x: x["elo"]["mu"], reverse=True)

    _write("team_ratings.json", {
        "updated": datetime.now(timezone.utc).isoformat(),
        "teams": teams_out,
    })


def export_model_weights(lr_weights: dict, xgb_info: dict, xgb_game_probs: dict) -> None:
    """Write docs/data/model_weights.json"""
    _write("model_weights.json", {
        "updated": datetime.now(timezone.utc).isoformat(),
        "logistic_regression": lr_weights,
        "xgboost": xgb_info,
        "xgb_game_probs": xgb_game_probs,
    })


def export_model_performance(metrics: dict) -> None:
    """Write docs/data/model_performance.json"""
    _write("model_performance.json", {
        "computed_on": datetime.now(timezone.utc).date().isoformat(),
        **metrics,
    })


def export_season_history(season_games: List[dict], team_elos_by_week: dict) -> None:
    """Write docs/data/season_history.json"""
    _write("season_history.json", {
        "updated": datetime.now(timezone.utc).isoformat(),
        "games": season_games,
    })


def export_odds(odds_data: List[dict]) -> None:
    """Write docs/data/odds.json"""
    _write("odds.json", {
        "updated": datetime.now(timezone.utc).isoformat(),
        "games": odds_data,
    })
