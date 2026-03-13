"""
bayesian_model.py
System 3: Bayesian updating of team strength distributions.
Models each team's rating as Normal(mu, sigma).
Sigma shrinks as more games are played, representing growing confidence.
"""

import logging
import math
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)

PRIOR_SIGMA = 75.0        # High uncertainty at season start
MIN_SIGMA = 15.0          # Floor — always some uncertainty
K_BAYESIAN = 0.85         # How quickly sigma shrinks (per game played)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_posteriors(elo_dict: Dict[str, float]) -> Dict[str, dict]:
    """
    Initialize Bayesian posteriors from current ELO ratings.
    mu = current ELO, sigma = PRIOR_SIGMA (high uncertainty at season start).
    """
    posteriors = {}
    for team, elo in elo_dict.items():
        posteriors[team] = {
            "mu": float(elo),
            "sigma": PRIOR_SIGMA,
            "games_played": 0,
        }
    return posteriors


# ---------------------------------------------------------------------------
# Bayesian update (conjugate normal-normal)
# ---------------------------------------------------------------------------

def bayesian_update(
    prior_mu: float,
    prior_sigma: float,
    observed_performance: float,
    observation_noise: float = 150.0,
    game_weight: float = 1.0,
) -> tuple:
    """
    Conjugate normal-normal Bayesian update.
    P(theta | data) = P(data | theta) * P(theta) / P(data)

    observed_performance: estimated true strength from this game
                          (e.g., ELO-implied rating from game result)
    observation_noise: how noisy a single game observation is (sigma of likelihood)
    game_weight: exponential decay weight for this game

    Returns (posterior_mu, posterior_sigma).
    """
    # Precision-weighted update
    prior_precision = 1.0 / (prior_sigma ** 2)
    likelihood_precision = game_weight / (observation_noise ** 2)

    posterior_precision = prior_precision + likelihood_precision
    posterior_sigma = math.sqrt(1.0 / posterior_precision)
    posterior_mu = (prior_mu * prior_precision + observed_performance * likelihood_precision) / posterior_precision

    # Enforce minimum sigma
    posterior_sigma = max(posterior_sigma, MIN_SIGMA)

    return posterior_mu, posterior_sigma


def update_team_posterior(
    posterior: dict,
    game_result: dict,
    elo_update: float,
    lambda_decay: float = 0.05,
    week: int = 1,
) -> dict:
    """
    Update a team's posterior after one game.

    game_result: dict with keys: result (1/0.5/0), point_diff, week
    elo_update: how much the ELO changed this game (used as signal strength)
    """
    games_played = posterior["games_played"] + 1
    live_weight = games_played / 17.0

    # Game decay weight
    decay_weight = math.exp(-lambda_decay * (17 - week))

    # Observed performance: current mu adjusted by ELO signal
    observed = posterior["mu"] + elo_update

    new_mu, new_sigma = bayesian_update(
        posterior["mu"],
        posterior["sigma"],
        observed,
        observation_noise=150.0 * (1 - live_weight * 0.5),  # noise shrinks with more data
        game_weight=decay_weight,
    )

    return {
        "mu": new_mu,
        "sigma": new_sigma,
        "games_played": games_played,
    }


# ---------------------------------------------------------------------------
# Process full season
# ---------------------------------------------------------------------------

def build_bayesian_posteriors(
    season_games: List[dict],
    starting_elos: Dict[str, float],
    lambda_decay: float = 0.05,
) -> Dict[str, dict]:
    """
    Walk through the current season's completed games and build posteriors.
    Returns dict of team -> {mu, sigma, games_played}.
    """
    log.info("Building Bayesian posteriors from season games...")
    posteriors = initialize_posteriors(starting_elos)

    for game in season_games:
        home = game["home_team"]
        away = game["away_team"]
        hs = game.get("home_score", 0)
        aws = game.get("away_score", 0)
        week = game.get("week", 1)

        if hs == 0 and aws == 0:
            continue

        point_diff = hs - aws
        home_result = 1.0 if hs > aws else (0.5 if hs == aws else 0.0)
        away_result = 1.0 - home_result

        # ELO-implied update (how surprising was this result?)
        from elo_model import calculate_expected
        home_prior = posteriors.get(home, {"mu": 1500.0, "sigma": PRIOR_SIGMA, "games_played": 0})
        away_prior = posteriors.get(away, {"mu": 1500.0, "sigma": PRIOR_SIGMA, "games_played": 0})
        expected_home = calculate_expected(home_prior["mu"], away_prior["mu"], hfa=65)
        home_surprise = (home_result - expected_home) * 100  # ELO-scale signal

        posteriors[home] = update_team_posterior(
            home_prior, {"result": home_result, "point_diff": point_diff, "week": week},
            elo_update=home_surprise, lambda_decay=lambda_decay, week=week,
        )
        posteriors[away] = update_team_posterior(
            away_prior, {"result": away_result, "point_diff": -point_diff, "week": week},
            elo_update=-home_surprise, lambda_decay=lambda_decay, week=week,
        )

    log.info(f"Bayesian posteriors: {len(posteriors)} teams")
    return posteriors


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_live_weight(games_played: int, total_games: int = 17) -> float:
    """Posterior weight — leans on observed data more as season progresses."""
    return min(1.0, games_played / total_games)
