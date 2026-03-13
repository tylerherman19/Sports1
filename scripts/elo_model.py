"""
elo_model.py
System 2: ELO Rating with MOV multiplier, recent form blend,
head-to-head adjustment, and injury discounts.
"""

import math
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_ELO = 1500
SEASON_REGRESSION_FACTOR = 0.75
REGRESSION_MEAN = 1505


# ---------------------------------------------------------------------------
# Season initialization
# ---------------------------------------------------------------------------

def initialize_elos(prior_season_final: Dict[str, float]) -> Dict[str, float]:
    """
    Apply mean-regression at season start.
    Season_start_ELO = Prior_final_ELO * 0.75 + 1505 * 0.25
    """
    return {
        team: prior_elo * SEASON_REGRESSION_FACTOR + REGRESSION_MEAN * (1 - SEASON_REGRESSION_FACTOR)
        for team, prior_elo in prior_season_final.items()
    }


# ---------------------------------------------------------------------------
# Core ELO math
# ---------------------------------------------------------------------------

def calculate_expected(elo_team: float, elo_opponent: float, hfa: float = 0.0) -> float:
    """
    Expected win probability for the team (not adjusted for home field here).
    hfa is added to elo_team before computing.
    """
    adjusted = elo_team + hfa
    return 1.0 / (1.0 + 10.0 ** ((elo_opponent - adjusted) / 400.0))


def calculate_mov_multiplier(point_diff: float, elo_diff: float) -> float:
    """
    Margin of victory multiplier.
    MOV_mult = log(abs(point_diff) + 1) * (2.2 / (ELO_diff * 0.001 + 2.2))
    """
    if point_diff == 0:
        point_diff = 1  # avoid log(1) = 0
    return math.log(abs(point_diff) + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))


def update_elo(
    elo_team: float,
    elo_opponent: float,
    result: float,  # 1.0 = win, 0.5 = tie, 0.0 = loss
    point_diff: float,
    k: float = 20.0,
    hfa: float = 0.0,
) -> float:
    """
    Return the new ELO for the team after a game.
    Elo_new = Elo_old + K * MOV_mult * (Actual - Expected)
    """
    expected = calculate_expected(elo_team, elo_opponent, hfa=hfa)
    elo_diff = elo_team + hfa - elo_opponent
    mov = calculate_mov_multiplier(point_diff if result >= 0.5 else -point_diff, elo_diff)
    delta = k * mov * (result - expected)
    return elo_team + delta


# ---------------------------------------------------------------------------
# Build full ELO history from FiveThirtyEight data
# ---------------------------------------------------------------------------

def build_elo_from_fte(fte_df: pd.DataFrame, k: float = 20.0, hfa: float = 65.0) -> Dict[str, float]:
    """
    Replay the entire FiveThirtyEight history to produce current team ELOs.
    Uses season-start regression between seasons.
    Returns dict of team -> current ELO.
    """
    log.info("Building ELO ratings from FTE historical data...")
    elos: Dict[str, float] = defaultdict(lambda: DEFAULT_ELO)
    current_season = None

    for _, row in fte_df.sort_values("date").iterrows():
        season = int(row["season"])
        team1 = row["team1"]
        team2 = row["team2"]
        score1 = float(row["score1"])
        score2 = float(row["score2"])

        # Apply season regression on season change
        if current_season is not None and season != current_season:
            elos = defaultdict(lambda: DEFAULT_ELO, initialize_elos(dict(elos)))
        current_season = season

        point_diff = score1 - score2
        result1 = 1.0 if score1 > score2 else (0.5 if score1 == score2 else 0.0)
        result2 = 1.0 - result1

        # team1 is home team in FTE data
        elo1 = elos[team1]
        elo2 = elos[team2]

        elos[team1] = update_elo(elo1, elo2, result1, point_diff, k=k, hfa=hfa)
        elos[team2] = update_elo(elo2, elo1, result2, -point_diff, k=k, hfa=0.0)

    return dict(elos)


# ---------------------------------------------------------------------------
# Current season ELO from game-by-game records
# ---------------------------------------------------------------------------

def build_season_elos(
    season_games: List[dict],
    starting_elos: Dict[str, float],
    k: float = 20.0,
    hfa: float = 65.0,
) -> Dict[str, float]:
    """
    Update ELOs through current season's completed games.
    season_games: list of dicts with keys: home_team, away_team, home_score, away_score
    Returns updated ELO dict.
    """
    elos = dict(starting_elos)
    for game in season_games:
        home = game["home_team"]
        away = game["away_team"]
        hs = game["home_score"]
        aws = game["away_score"]
        if hs == 0 and aws == 0:
            continue  # unplayed
        point_diff = hs - aws
        result_home = 1.0 if hs > aws else (0.5 if hs == aws else 0.0)
        result_away = 1.0 - result_home
        elo_h = elos.get(home, DEFAULT_ELO)
        elo_a = elos.get(away, DEFAULT_ELO)
        elos[home] = update_elo(elo_h, elo_a, result_home, point_diff, k=k, hfa=hfa)
        elos[away] = update_elo(elo_a, elo_h, result_away, -point_diff, k=k, hfa=0.0)
    return elos


# ---------------------------------------------------------------------------
# Recent form ELO (last N games)
# ---------------------------------------------------------------------------

def calculate_recent_form_elo(
    team: str,
    season_games: List[dict],
    k: float = 20.0,
    hfa: float = 65.0,
    last_n: int = 5,
) -> float:
    """
    Calculate a separate ELO rating using only the last N games for a team.
    """
    team_games = [
        g for g in season_games
        if (g["home_team"] == team or g["away_team"] == team)
        and (g["home_score"] > 0 or g["away_score"] > 0)
    ]
    recent = team_games[-last_n:]
    form_elo = DEFAULT_ELO
    for game in recent:
        is_home = game["home_team"] == team
        opponent = game["away_team"] if is_home else game["home_team"]
        opp_elo = DEFAULT_ELO  # simplified — use league average for form calc
        if is_home:
            pd_val = game["home_score"] - game["away_score"]
            result = 1.0 if pd_val > 0 else (0.5 if pd_val == 0 else 0.0)
        else:
            pd_val = game["away_score"] - game["home_score"]
            result = 1.0 if pd_val > 0 else (0.5 if pd_val == 0 else 0.0)
        form_elo = update_elo(form_elo, opp_elo, result, pd_val, k=k, hfa=hfa if is_home else 0.0)
    return form_elo


def blend_form_elo(full_elo: float, form_elo: float, blend: float = 0.30) -> float:
    """
    Blend season ELO with recent form ELO.
    blend=0.30 means 30% recent form, 70% full season.
    """
    return full_elo * (1 - blend) + form_elo * blend


# ---------------------------------------------------------------------------
# Head-to-head adjustment
# ---------------------------------------------------------------------------

def calculate_h2h_adjustment(
    team_a: str,
    team_b: str,
    all_games: List[dict],
    weight: float = 0.20,
    last_n: int = 10,
    max_adjustment: float = 50.0,
) -> float:
    """
    Returns ELO adjustment for team_a vs team_b based on H2H history.
    Positive value favors team_a.
    """
    h2h = [
        g for g in all_games
        if (g["home_team"] in {team_a, team_b} and g["away_team"] in {team_a, team_b})
        and (g["home_score"] > 0 or g["away_score"] > 0)
    ]
    recent_h2h = h2h[-last_n:]
    if not recent_h2h:
        return 0.0
    a_wins = sum(
        1 for g in recent_h2h
        if (g["home_team"] == team_a and g["home_score"] > g["away_score"])
        or (g["away_team"] == team_a and g["away_score"] > g["home_score"])
    )
    a_win_rate = a_wins / len(recent_h2h)
    raw_adjustment = (a_win_rate - 0.5) * 2 * max_adjustment
    return raw_adjustment * weight


# ---------------------------------------------------------------------------
# Injury discount
# ---------------------------------------------------------------------------

def apply_injury_discount(elo: float, discount: float) -> float:
    """Subtract injury discount directly from ELO (0–200 points)."""
    return elo - max(0.0, float(discount))


# ---------------------------------------------------------------------------
# Pre-game ELO with all adjustments applied
# ---------------------------------------------------------------------------

def pregame_elo(
    home_team: str,
    away_team: str,
    base_elos: Dict[str, float],
    season_games: List[dict],
    all_games: List[dict],
    sliders: Optional[dict] = None,
) -> dict:
    """
    Compute pre-game adjusted ELOs for both teams, applying:
      - Recent form blend
      - Head-to-head adjustment
      - Home field advantage
      - Injury discounts

    Returns dict with keys: home_elo, away_elo, home_elo_adj, away_elo_adj,
    elo_diff (home minus away, post-HFA)
    """
    if sliders is None:
        sliders = {}
    k = float(sliders.get("k_factor", 20))
    hfa = float(sliders.get("home_field", 65))
    form_blend = float(sliders.get("form_blend", 0.30))
    h2h_weight = float(sliders.get("h2h_weight", 0.20))
    home_injury = float(sliders.get(f"injury_{home_team}", 0))
    away_injury = float(sliders.get(f"injury_{away_team}", 0))

    home_full = base_elos.get(home_team, DEFAULT_ELO)
    away_full = base_elos.get(away_team, DEFAULT_ELO)

    home_form = calculate_recent_form_elo(home_team, season_games, k=k, hfa=hfa)
    away_form = calculate_recent_form_elo(away_team, season_games, k=k, hfa=hfa)

    home_blended = blend_form_elo(home_full, home_form, blend=form_blend)
    away_blended = blend_form_elo(away_full, away_form, blend=form_blend)

    h2h_adj = calculate_h2h_adjustment(home_team, away_team, all_games, weight=h2h_weight)

    home_adj = apply_injury_discount(home_blended + h2h_adj, home_injury)
    away_adj = apply_injury_discount(away_blended - h2h_adj, away_injury)

    elo_diff = (home_adj + hfa) - away_adj

    return {
        "home_elo": home_full,
        "away_elo": away_full,
        "home_elo_adj": home_adj,
        "away_elo_adj": away_adj,
        "home_form_elo": home_form,
        "away_form_elo": away_form,
        "h2h_adj": h2h_adj,
        "elo_diff": elo_diff,
        "elo_prob": calculate_expected(home_adj, away_adj, hfa=hfa),
    }
