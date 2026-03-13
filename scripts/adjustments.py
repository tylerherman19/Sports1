"""
adjustments.py
System 6: Turnover Regression (regress toward mean as season progresses)
System 7: Rest and Travel Adjustments (rest days, distance, bye week, back-to-back)
System 8: Time Series Decay (exponential decay weights for training)
"""

import math
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System 6: Turnover Regression
# ---------------------------------------------------------------------------

ELO_PER_TURNOVER = 8.0  # each adjusted turnover = ±8 ELO points


def turnover_adjustment(
    actual_to_diff: float,
    games_played: int,
    total_games: int = 17,
    turnover_scale: float = 1.0,
) -> float:
    """
    Regress turnover differential toward zero as season progresses.
    Regression_weight = 1 - (games_played / 17)
    Adj_TO_diff = Actual_TO_diff * (1 - Regression_weight)
               = Actual_TO_diff * (games_played / 17)
    Returns ELO adjustment (positive means home team benefits).
    """
    regression_weight = 1.0 - (games_played / total_games)
    adj_to_diff = actual_to_diff * (1.0 - regression_weight)
    return adj_to_diff * ELO_PER_TURNOVER * turnover_scale


def compute_turnover_diffs(
    pfr_stats: Dict[str, dict],
    standings: Dict[str, dict],
) -> Dict[str, float]:
    """
    Compute net turnover differential per team (turnovers forced minus committed).
    Returns dict of team -> TO_diff.
    """
    to_diffs = {}
    for team, stats in pfr_stats.items():
        forced = float(stats.get("turnovers_forced", 0))
        committed = float(stats.get("turnovers_committed", 0))
        to_diffs[team] = forced - committed
    return to_diffs


# ---------------------------------------------------------------------------
# System 7: Rest and Travel Adjustments
# ---------------------------------------------------------------------------

REST_SCALE_PER_DAY = 1.5    # ELO points per extra day of rest (beyond 7)
SHORT_WEEK_THRESHOLD = 6    # days
SHORT_WEEK_PENALTY = -10.0
BYE_WEEK_BONUS = 25.0
BACK_TO_BACK_AWAY_PENALTY = -15.0
TRAVEL_SCALE = -3.0         # ELO points per 1000 miles

# NFL team city coordinates (lat, lon) for distance calculation
TEAM_COORDS = {
    "ARI": (33.5277, -112.2626),
    "ATL": (33.7550, -84.4010),
    "BAL": (39.2781, -76.6227),
    "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8531),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456),
    "GB":  (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373),
    "KC":  (39.0489, -94.4839),
    "LAC": (33.8644, -118.2611),
    "LAR": (33.9534, -118.3393),
    "LV":  (36.0909, -115.1833),
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9736, -93.2575),
    "NE":  (42.0909, -71.2643),
    "NO":  (29.9511, -90.0812),
    "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745),
    "PHI": (39.9007, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316),
    "SF":  (37.4033, -121.9694),
    "TB":  (27.9759, -82.5033),
    "TEN": (36.1664, -86.7713),
    "WAS": (38.9076, -76.8645),
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two lat/lon points in miles."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def travel_distance(team: str, opponent_city: str) -> float:
    """
    Calculate travel distance for a team traveling to opponent's city.
    Returns distance in miles (0 for home team).
    """
    if team not in TEAM_COORDS or opponent_city not in TEAM_COORDS:
        return 0.0
    lat1, lon1 = TEAM_COORDS[team]
    lat2, lon2 = TEAM_COORDS[opponent_city]
    return haversine_miles(lat1, lon1, lat2, lon2)


def rest_adjustment(
    days_rest: int,
    rest_travel_scale: float = 1.0,
    is_bye_coming_off: bool = False,
    is_back_to_back_away: bool = False,
) -> float:
    """
    Calculate rest-based ELO adjustment for a team.
    Positive = benefit, negative = penalty.
    """
    if is_bye_coming_off:
        return BYE_WEEK_BONUS * rest_travel_scale

    adjustment = 0.0
    if days_rest < SHORT_WEEK_THRESHOLD:
        adjustment += SHORT_WEEK_PENALTY

    # Extra/fewer days vs average 7
    adjustment += (days_rest - 7) * REST_SCALE_PER_DAY

    if is_back_to_back_away:
        adjustment += BACK_TO_BACK_AWAY_PENALTY

    return adjustment * rest_travel_scale


def travel_penalty(
    distance_miles: float,
    rest_travel_scale: float = 1.0,
) -> float:
    """
    Calculate travel distance ELO penalty.
    Travel_penalty = (distance_miles / 1000) * TRAVEL_SCALE
    """
    return (distance_miles / 1000.0) * TRAVEL_SCALE * rest_travel_scale


def compute_game_adjustments(
    home_team: str,
    away_team: str,
    home_rest_days: int,
    away_rest_days: int,
    home_prev_location: Optional[str] = None,
    away_prev_location: Optional[str] = None,
    home_off_bye: bool = False,
    away_off_bye: bool = False,
    rest_travel_scale: float = 1.0,
    turnover_scale: float = 1.0,
    home_to_diff: float = 0.0,
    away_to_diff: float = 0.0,
    home_games_played: int = 9,
    away_games_played: int = 9,
) -> dict:
    """
    Compute all rest, travel, and turnover ELO adjustments for a matchup.
    Returns dict of adjustment components.
    """
    # Rest
    home_rest_adj = rest_adjustment(home_rest_days, rest_travel_scale, is_bye_coming_off=home_off_bye)
    away_rest_adj = rest_adjustment(away_rest_days, rest_travel_scale, is_bye_coming_off=away_off_bye)

    # Travel (away team always travels; home team travels 0)
    away_travel_dist = travel_distance(away_team, home_team)
    home_travel_dist = 0.0
    away_travel_pen = travel_penalty(away_travel_dist, rest_travel_scale)
    home_travel_pen = 0.0

    # Back-to-back away check
    is_btb_away = (away_prev_location is not None and away_prev_location != away_team)
    if is_btb_away:
        away_rest_adj += BACK_TO_BACK_AWAY_PENALTY * rest_travel_scale

    # Turnover adjustments
    home_to_adj = turnover_adjustment(home_to_diff, home_games_played, turnover_scale=turnover_scale)
    away_to_adj = turnover_adjustment(away_to_diff, away_games_played, turnover_scale=turnover_scale)

    return {
        "home_rest_adj": round(home_rest_adj, 2),
        "away_rest_adj": round(away_rest_adj, 2),
        "home_travel_dist": round(home_travel_dist, 1),
        "away_travel_dist": round(away_travel_dist, 1),
        "home_travel_pen": round(home_travel_pen, 2),
        "away_travel_pen": round(away_travel_pen, 2),
        "home_to_adj": round(home_to_adj, 2),
        "away_to_adj": round(away_to_adj, 2),
        "net_adj": round(
            (home_rest_adj - away_rest_adj) +
            (home_travel_pen - away_travel_pen) +
            (home_to_adj - away_to_adj), 2
        ),
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
    }


# ---------------------------------------------------------------------------
# System 8: Time Series Decay
# ---------------------------------------------------------------------------

def decay_weight(weeks_ago: float, lambda_val: float = 0.05) -> float:
    """
    Exponential decay weight for historical game.
    weight = e^(-lambda * t) where t = weeks ago
    """
    return math.exp(-lambda_val * weeks_ago)


def apply_decay_to_games(
    games: List[dict],
    lambda_val: float = 0.05,
    reference_week: int = 18,
) -> List[dict]:
    """
    Attach a decay weight to each game based on how many weeks ago it was played.
    Mutates games in place, adding 'decay_weight' key.
    """
    for game in games:
        week = game.get("week", reference_week)
        weeks_ago = max(0, reference_week - week)
        game["decay_weight"] = decay_weight(weeks_ago, lambda_val)
    return games
