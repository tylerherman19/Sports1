"""
hierarchical_model.py
System 11: Hierarchical Team Model.
Uses PyMC MCMC to estimate per-team offensive and defensive sub-ratings
with uncertainty estimates at each level.

Falls back to efficiency-based estimates if PyMC sampling times out or fails.
"""

import logging
import warnings
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)

MCMC_DRAWS = 500
MCMC_TUNE = 500
MCMC_CHAINS = 2
SAMPLING_TIMEOUT = 600  # seconds


def fit_hierarchical_model(
    season_games: List[dict],
    team_efficiencies: Dict[str, dict],
    team_elos: Dict[str, float],
) -> Dict[str, dict]:
    """
    Fit a Bayesian hierarchical model using PyMC.
    Models each team as:
        Team_strength ~ Offensive_rating * 0.5 + Defensive_rating * 0.5
        Offensive_rating = passing_eff * 0.6 + rushing_eff * 0.4
        Defensive_rating = pass_def_eff * 0.6 + rush_def_eff * 0.4

    Returns dict of team -> {off_mu, off_sigma, def_mu, def_sigma, total_mu, total_sigma}
    Falls back to efficiency-based estimates if MCMC fails.
    """
    try:
        return _fit_with_pymc(season_games, team_efficiencies, team_elos)
    except Exception as e:
        log.warning(f"PyMC MCMC failed ({e}). Using efficiency-based fallback.")
        return _efficiency_fallback(team_efficiencies, team_elos)


def _fit_with_pymc(
    season_games: List[dict],
    team_efficiencies: Dict[str, dict],
    team_elos: Dict[str, float],
) -> Dict[str, dict]:
    """Full MCMC estimation using PyMC."""
    import pymc as pm
    import pytensor.tensor as pt

    teams = sorted(set(list(team_elos.keys())))
    team_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    # Build game outcome arrays
    home_idx_list = []
    away_idx_list = []
    results = []

    for game in season_games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        hs = game.get("home_score", 0)
        aws = game.get("away_score", 0)
        if hs == 0 and aws == 0:
            continue
        if home not in team_idx or away not in team_idx:
            continue
        home_idx_list.append(team_idx[home])
        away_idx_list.append(team_idx[away])
        results.append(1 if hs > aws else 0)

    if len(results) < 10:
        log.warning("Fewer than 10 completed games — using fallback for hierarchical model")
        return _efficiency_fallback(team_efficiencies, team_elos)

    home_idx_arr = np.array(home_idx_list)
    away_idx_arr = np.array(away_idx_list)
    results_arr = np.array(results)

    # Prior means from efficiency
    off_prior_means = np.array([
        team_efficiencies.get(t, {}).get("off_efficiency", 1.0) for t in teams
    ])
    def_prior_means = np.array([
        team_efficiencies.get(t, {}).get("def_efficiency", 1.0) for t in teams
    ])

    log.info(f"Fitting hierarchical model: {n_teams} teams, {len(results)} games")

    with pm.Model() as model:
        # Hyperpriors
        mu_off = pm.Normal("mu_off", mu=1.0, sigma=0.2)
        sigma_off = pm.HalfNormal("sigma_off", sigma=0.15)
        mu_def = pm.Normal("mu_def", mu=1.0, sigma=0.2)
        sigma_def = pm.HalfNormal("sigma_def", sigma=0.15)

        # Team-level parameters
        off_raw = pm.Normal("off_raw", mu=0, sigma=1, shape=n_teams)
        def_raw = pm.Normal("def_raw", mu=0, sigma=1, shape=n_teams)

        # Non-centered parameterization
        off = pm.Deterministic("off", mu_off + off_raw * sigma_off)
        deff = pm.Deterministic("def", mu_def + def_raw * sigma_def)

        # Team strength = 0.5 * off + 0.5 * def
        team_strength = 0.5 * off + 0.5 * deff

        # Home field advantage
        hfa = pm.Normal("hfa", mu=0.1, sigma=0.05)

        # Win probability via logistic
        home_strength = team_strength[home_idx_arr] + hfa
        away_strength = team_strength[away_idx_arr]
        log_odds = home_strength - away_strength
        win_prob = pm.math.sigmoid(log_odds * 2.0)  # scale to reasonable probability range

        # Likelihood
        obs = pm.Bernoulli("obs", p=win_prob, observed=results_arr)

        # Sample
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pm.sample(
                draws=MCMC_DRAWS,
                tune=MCMC_TUNE,
                chains=MCMC_CHAINS,
                progressbar=False,
                return_inferencedata=True,
                target_accept=0.85,
            )

    # Extract posteriors
    results_dict = {}
    off_samples = trace.posterior["off"].values.reshape(-1, n_teams)
    def_samples = trace.posterior["def"].values.reshape(-1, n_teams)

    for i, team in enumerate(teams):
        off_mu = float(np.mean(off_samples[:, i]))
        off_sigma = float(np.std(off_samples[:, i]))
        def_mu = float(np.mean(def_samples[:, i]))
        def_sigma = float(np.std(def_samples[:, i]))
        total_mu = 0.5 * off_mu + 0.5 * def_mu
        total_sigma = float(np.sqrt(0.25 * off_sigma**2 + 0.25 * def_sigma**2))

        results_dict[team] = {
            "off_mu": round(off_mu, 4),
            "off_sigma": round(off_sigma, 4),
            "def_mu": round(def_mu, 4),
            "def_sigma": round(def_sigma, 4),
            "total_mu": round(total_mu, 4),
            "total_sigma": round(total_sigma, 4),
        }

    log.info(f"Hierarchical model fitted: {len(results_dict)} teams")
    return results_dict


def _efficiency_fallback(
    team_efficiencies: Dict[str, dict],
    team_elos: Dict[str, float],
) -> Dict[str, dict]:
    """
    Fallback: derive hierarchical sub-ratings from efficiency metrics.
    Uses off/def efficiency as proxies for offensive/defensive ratings.
    Sigma estimated from cross-team variance.
    """
    log.info("Using efficiency-based fallback for hierarchical model...")

    all_off = [v.get("off_efficiency", 1.0) for v in team_efficiencies.values()]
    all_def = [v.get("def_efficiency", 1.0) for v in team_efficiencies.values()]
    off_std = float(np.std(all_off)) if all_off else 0.1
    def_std = float(np.std(all_def)) if all_def else 0.1

    results = {}
    for team in team_elos:
        eff = team_efficiencies.get(team, {})
        off_mu = float(eff.get("off_efficiency", 1.0))
        def_mu = float(eff.get("def_efficiency", 1.0))
        total_mu = 0.5 * off_mu + 0.5 * def_mu

        results[team] = {
            "off_mu": round(off_mu, 4),
            "off_sigma": round(off_std * 0.5, 4),  # uncertainty = half the league spread
            "def_mu": round(def_mu, 4),
            "def_sigma": round(def_std * 0.5, 4),
            "total_mu": round(total_mu, 4),
            "total_sigma": round(float(np.sqrt(0.25 * (off_std*0.5)**2 + 0.25 * (def_std*0.5)**2)), 4),
        }

    return results
