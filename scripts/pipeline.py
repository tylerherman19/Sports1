"""
pipeline.py
Main orchestrator. Runs the complete data pipeline:
1. Fetch all data (FTE, ESPN, Odds API, PFR)
2. Build ELO ratings
3. Train ML models
4. Build Bayesian posteriors
5. Compute Pythagorean, efficiency, adjustments
6. Fit hierarchical model
7. Run Monte Carlo season simulation
8. Compute model metrics
9. Export all JSON files

Run via: python scripts/pipeline.py
Or triggered by GitHub Actions workflow.
"""

import os
import sys
import logging
import math
from collections import defaultdict
from typing import Dict, List
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def run_pipeline():
    log.info("=" * 60)
    log.info("NFL PREDICTION PIPELINE STARTING")
    log.info("=" * 60)

    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    # -----------------------------------------------------------------------
    # 1. DATA FETCHING
    # -----------------------------------------------------------------------
    log.info("STEP 1: Fetching data...")
    from data_fetcher import (
        fetch_fte_historical,
        fetch_espn_scoreboard,
        fetch_espn_standings,
        fetch_espn_injuries,
        fetch_odds,
        scrape_pfr_efficiency,
        calculate_rest_days,
        american_to_implied,
    )

    fte_df = fetch_fte_historical()
    espn_games = fetch_espn_scoreboard()
    standings = fetch_espn_standings()
    injuries = fetch_espn_injuries()
    odds_data = fetch_odds(odds_api_key)
    pfr_stats = scrape_pfr_efficiency()

    # Current season: filter ESPN games that are completed
    season_games = [
        g for g in espn_games
        if g.get("status") == "STATUS_FINAL"
        and (g["home_score"] > 0 or g["away_score"] > 0)
    ]

    # Upcoming games
    upcoming_games = [
        g for g in espn_games
        if g.get("status") != "STATUS_FINAL"
    ]

    log.info(f"Season games completed: {len(season_games)}, upcoming: {len(upcoming_games)}")

    # -----------------------------------------------------------------------
    # 2. ELO RATINGS
    # -----------------------------------------------------------------------
    log.info("STEP 2: Building ELO ratings...")
    from elo_model import build_elo_from_fte, initialize_elos, build_season_elos, DEFAULT_ELO

    # Build end-of-prior-season ELOs from FTE historical
    prior_season = int(fte_df["season"].max()) - 1
    prior_fte = fte_df[fte_df["season"] <= prior_season]
    prior_elos = build_elo_from_fte(prior_fte, k=20.0, hfa=65.0)

    # Season-start regression
    season_start_elos = initialize_elos(prior_elos)

    # Update through current season games
    current_elos = build_season_elos(season_games, season_start_elos, k=20.0, hfa=65.0)

    # Ensure all 32 teams exist
    from teams_config import NFL_TEAMS
    for abbrev in NFL_TEAMS:
        if abbrev not in current_elos:
            current_elos[abbrev] = DEFAULT_ELO

    log.info(f"ELO computed for {len(current_elos)} teams. Top 5:")
    top5 = sorted(current_elos.items(), key=lambda x: x[1], reverse=True)[:5]
    for team, elo in top5:
        log.info(f"  {team}: {elo:.1f}")

    # -----------------------------------------------------------------------
    # 3. ML MODEL TRAINING
    # -----------------------------------------------------------------------
    log.info("STEP 3: Training ML models...")
    from ml_models import (
        build_feature_matrix,
        train_logistic_regression,
        train_xgboost,
        lr_predict,
        xgb_predict,
        export_xgb_game_probs,
        FEATURE_NAMES_LR,
    )

    lambda_decay = 0.05
    X_train, y_train, w_train = build_feature_matrix(fte_df, pfr_stats, lambda_decay=lambda_decay)

    lr_model, lr_weights = train_logistic_regression(X_train, y_train, w_train)
    xgb_model, xgb_info = train_xgboost(X_train, y_train, w_train, feature_names=FEATURE_NAMES_LR)

    log.info("ML models trained successfully")

    # -----------------------------------------------------------------------
    # 4. BAYESIAN POSTERIORS
    # -----------------------------------------------------------------------
    log.info("STEP 4: Building Bayesian posteriors...")
    from bayesian_model import build_bayesian_posteriors

    bayesian_posteriors = build_bayesian_posteriors(
        season_games,
        current_elos,
        lambda_decay=lambda_decay,
    )

    # -----------------------------------------------------------------------
    # 5. PYTHAGOREAN + EFFICIENCY + ADJUSTMENTS
    # -----------------------------------------------------------------------
    log.info("STEP 5: Computing Pythagorean, efficiency, and adjustments...")
    from pythagorean import compute_team_pythagorean
    from efficiency import compute_team_efficiencies
    from adjustments import (
        compute_game_adjustments,
        compute_turnover_diffs,
        apply_decay_to_games,
    )

    pythagorean_data = compute_team_pythagorean(standings)
    efficiency_data = compute_team_efficiencies(pfr_stats, current_elos, season_games)
    to_diffs = compute_turnover_diffs(pfr_stats, standings)

    # Apply decay weights to season games
    season_games_weighted = apply_decay_to_games(list(season_games), lambda_decay=lambda_decay)

    # Compute adjustments for each upcoming game
    adjustments_data = {}
    for game in espn_games:
        home = game["home_team"]
        away = game["away_team"]
        game_key = f"{home}_vs_{away}"

        home_rest = calculate_rest_days(game.get("home_team_id", home))
        away_rest = calculate_rest_days(game.get("away_team_id", away))

        home_gp = standings.get(home, {}).get("games_played", 9)
        away_gp = standings.get(away, {}).get("games_played", 9)

        adjs = compute_game_adjustments(
            home_team=home,
            away_team=away,
            home_rest_days=home_rest,
            away_rest_days=away_rest,
            rest_travel_scale=1.0,
            turnover_scale=1.0,
            home_to_diff=to_diffs.get(home, 0.0),
            away_to_diff=to_diffs.get(away, 0.0),
            home_games_played=home_gp,
            away_games_played=away_gp,
        )
        adjustments_data[game_key] = adjs

    # -----------------------------------------------------------------------
    # 6. HIERARCHICAL MODEL
    # -----------------------------------------------------------------------
    log.info("STEP 6: Fitting hierarchical model...")
    from hierarchical_model import fit_hierarchical_model

    hierarchical_data = fit_hierarchical_model(season_games, efficiency_data, current_elos)

    # -----------------------------------------------------------------------
    # 7. MONTE CARLO SEASON SIMULATION
    # -----------------------------------------------------------------------
    log.info("STEP 7: Running Monte Carlo playoff simulation...")
    monte_carlo_playoffs = run_monte_carlo_season(
        current_elos, bayesian_posteriors, standings
    )

    # -----------------------------------------------------------------------
    # 8. XGBoost game probabilities for current games
    # -----------------------------------------------------------------------
    log.info("STEP 8: Computing XGBoost game probabilities...")
    # Build feature rows for current games
    current_game_features = []
    for game in espn_games:
        home = game["home_team"]
        away = game["away_team"]
        game_key = f"{home}_vs_{away}"
        adjs = adjustments_data.get(game_key, {})
        home_eff = efficiency_data.get(home, {})
        away_eff = efficiency_data.get(away, {})

        features = {
            "elo_diff": current_elos.get(home, DEFAULT_ELO) - current_elos.get(away, DEFAULT_ELO),
            "offensive_rating_diff": home_eff.get("off_efficiency", 1.0) - away_eff.get("off_efficiency", 1.0),
            "defensive_rating_diff": home_eff.get("def_efficiency", 1.0) - away_eff.get("def_efficiency", 1.0),
            "home_field": 1.0,
            "rest_days_diff": adjs.get("home_rest_days", 7) - adjs.get("away_rest_days", 7),
            "pythagorean_diff": (
                pythagorean_data.get(home, {}).get("expectation", 0.5) -
                pythagorean_data.get(away, {}).get("expectation", 0.5)
            ),
            "net_efficiency_diff": home_eff.get("net_efficiency", 0) - away_eff.get("net_efficiency", 0),
            "turnover_diff_adjusted": adjs.get("home_to_adj", 0) - adjs.get("away_to_adj", 0),
            "travel_distance_diff": adjs.get("away_travel_dist", 0),
            "last5_win_rate_diff": 0.0,
        }
        game["features"] = features

    xgb_game_probs = export_xgb_game_probs(xgb_model, espn_games, FEATURE_NAMES_LR)

    # -----------------------------------------------------------------------
    # 9. MODEL METRICS
    # -----------------------------------------------------------------------
    log.info("STEP 9: Computing model performance metrics...")
    from ensemble import compute_model_metrics

    metrics = compute_model_metrics(
        fte_df,
        lr_model,
        xgb_model,
        feature_names=FEATURE_NAMES_LR,
        lambda_decay=lambda_decay,
    )

    # -----------------------------------------------------------------------
    # 10. EXPORT JSON
    # -----------------------------------------------------------------------
    log.info("STEP 10: Exporting JSON files...")
    from export_json import (
        export_games,
        export_team_ratings,
        export_model_weights,
        export_model_performance,
        export_season_history,
        export_odds,
    )

    # Build team_elos dict with full info for export
    team_elos_full = {}
    for team in current_elos:
        posterior = bayesian_posteriors.get(team, {"mu": current_elos[team], "sigma": 75.0})
        team_elos_full[team] = {
            "mu": posterior["mu"],
            "sigma": posterior["sigma"],
            "elo_raw": current_elos[team],
            "elo_diff": 0.0,  # filled per-game
            "last5_win_rate": 0.5,
        }

    export_games(
        espn_games=espn_games,
        team_elos=team_elos_full,
        bayesian_posteriors=bayesian_posteriors,
        pythagorean_data=pythagorean_data,
        efficiency_data=efficiency_data,
        adjustments_data=adjustments_data,
        lr_weights=lr_weights,
        xgb_probs=xgb_game_probs,
        odds_data=odds_data,
        standings=standings,
    )

    export_team_ratings(
        team_elos=current_elos,
        bayesian_posteriors=bayesian_posteriors,
        pythagorean_data=pythagorean_data,
        efficiency_data=efficiency_data,
        hierarchical_data=hierarchical_data,
        standings=standings,
        season_games=season_games,
        monte_carlo_playoffs=monte_carlo_playoffs,
    )

    export_model_weights(lr_weights, xgb_info, xgb_game_probs)
    export_model_performance(metrics)
    export_season_history(season_games, {})
    export_odds(odds_data)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE — all JSON files written to docs/data/")
    log.info("=" * 60)


def run_monte_carlo_season(
    team_elos: Dict[str, float],
    bayesian_posteriors: Dict[str, dict],
    standings: Dict[str, dict],
    n_simulations: int = 10000,
) -> Dict[str, float]:
    """
    Simulate remaining season 10,000 times to estimate playoff probabilities.
    Simple model: teams above 0.5 win rate make playoffs.
    Returns dict of team -> playoff probability.
    """
    from elo_model import calculate_expected

    teams = list(team_elos.keys())
    playoff_counts = defaultdict(int)

    for _ in range(n_simulations):
        # Sample team strengths from Bayesian posteriors
        sim_elos = {}
        for team in teams:
            posterior = bayesian_posteriors.get(team, {"mu": team_elos.get(team, 1500), "sigma": 75})
            sampled = np.random.normal(posterior["mu"], posterior["sigma"])
            sim_elos[team] = sampled

        # Simple playoff determination: top 7 per conference by simulated ELO
        # We'll use a simplified all-teams ranking since we don't have schedule data
        sorted_teams = sorted(sim_elos.items(), key=lambda x: x[1], reverse=True)

        # Top 14 teams make playoffs (7 per conference, simplified)
        playoff_teams = [t for t, _ in sorted_teams[:14]]
        for team in playoff_teams:
            playoff_counts[team] += 1

    return {team: playoff_counts[team] / n_simulations for team in teams}


if __name__ == "__main__":
    run_pipeline()
