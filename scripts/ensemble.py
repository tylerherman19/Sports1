"""
ensemble.py
System 12: Ensemble blend of all prediction systems.
Also computes model performance metrics (log loss, Brier, AUC, calibration).
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
from scipy.special import expit  # sigmoid

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "lr": 0.30,
    "xgb": 0.25,
    "elo": 0.20,
    "pyth": 0.15,
    "eff": 0.10,
}


# ---------------------------------------------------------------------------
# Core probability conversions
# ---------------------------------------------------------------------------

def elo_to_prob(elo_diff: float) -> float:
    """Convert ELO difference (home - away) to win probability."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def prob_to_elo_diff(prob: float) -> float:
    """Convert win probability to equivalent ELO difference."""
    prob = max(0.001, min(0.999, prob))
    return -400.0 * math.log10(1.0 / prob - 1.0)


# ---------------------------------------------------------------------------
# Ensemble blend (System 12)
# ---------------------------------------------------------------------------

def compute_ensemble_prob(
    lr_prob: float,
    xgb_prob: float,
    elo_prob: float,
    pyth_prob: float,
    eff_prob: float,
    weights: Optional[dict] = None,
) -> float:
    """
    Weighted ensemble of all probability signals.
    Final_prob = lr*w1 + xgb*w2 + elo*w3 + pyth*w4 + eff*w5
    Weights are normalized to sum to 1.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    w = weights
    total_w = w.get("lr", 0.30) + w.get("xgb", 0.25) + w.get("elo", 0.20) + w.get("pyth", 0.15) + w.get("eff", 0.10)
    if total_w <= 0:
        total_w = 1.0
    prob = (
        lr_prob   * w.get("lr", 0.30) +
        xgb_prob  * w.get("xgb", 0.25) +
        elo_prob  * w.get("elo", 0.20) +
        pyth_prob * w.get("pyth", 0.15) +
        eff_prob  * w.get("eff", 0.10)
    ) / total_w
    return max(0.001, min(0.999, prob))


def apply_elo_modifiers(
    base_elo_diff: float,
    adjustments: dict,
    rest_travel_scale: float = 1.0,
    turnover_scale: float = 1.0,
) -> float:
    """
    Apply additive ELO modifiers (rest, travel, turnovers, H2H) to base ELO difference.
    Returns modified ELO diff for final probability conversion.
    """
    net = base_elo_diff
    net += adjustments.get("home_rest_adj", 0) - adjustments.get("away_rest_adj", 0)
    net += adjustments.get("home_travel_pen", 0) - adjustments.get("away_travel_pen", 0)
    net += adjustments.get("home_to_adj", 0) - adjustments.get("away_to_adj", 0)
    net += adjustments.get("h2h_adj", 0)
    return net


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

def kelly_fraction(
    model_prob: float,
    market_odds_american: float,
    max_kelly: float = 0.25,
) -> float:
    """
    Kelly Criterion bet sizing: f* = (b*p - q) / b
    b = decimal odds payout (how much you win per unit bet)
    p = model win probability
    q = 1 - p

    Returns fraction of bankroll to bet (capped at max_kelly for safety).
    Note: displayed for reference only, not investment advice.
    """
    if market_odds_american >= 0:
        b = market_odds_american / 100.0
    else:
        b = 100.0 / abs(market_odds_american)

    p = max(0.001, min(0.999, model_prob))
    q = 1.0 - p
    kelly = (b * p - q) / b

    return max(0.0, min(max_kelly, kelly))


def compute_edge(model_prob: float, market_implied_prob: float) -> float:
    """
    Edge = Model_prob - Market_prob
    Positive = model sees value, negative = market disagrees.
    """
    return model_prob - market_implied_prob


# ---------------------------------------------------------------------------
# Model performance metrics
# ---------------------------------------------------------------------------

def calculate_log_loss(y_true: List[int], y_pred: List[float]) -> float:
    """
    Log Loss = -(1/N) * sum(y*log(p) + (1-y)*log(1-p))
    Measures probability calibration quality.
    """
    eps = 1e-9
    n = len(y_true)
    if n == 0:
        return float("nan")
    total = 0.0
    for y, p in zip(y_true, y_pred):
        p = max(eps, min(1 - eps, p))
        total += y * math.log(p) + (1 - y) * math.log(1 - p)
    return -total / n


def calculate_brier_score(y_true: List[int], y_pred: List[float]) -> float:
    """
    Brier Score = (1/N) * sum((p - y)^2)
    Measures forecast accuracy (lower is better).
    """
    n = len(y_true)
    if n == 0:
        return float("nan")
    return sum((p - y) ** 2 for y, p in zip(y_true, y_pred)) / n


def calculate_auc(y_true: List[int], y_pred: List[float]) -> float:
    """
    AUC-ROC via trapezoidal rule.
    Measures discrimination ability (ability to rank winners above losers).
    """
    from sklearn.metrics import roc_auc_score
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def calibration_buckets(
    y_true: List[int],
    y_pred: List[float],
    buckets: Optional[List[tuple]] = None,
) -> List[dict]:
    """
    Group predictions into probability buckets and compute actual win rates.
    Default buckets: 50-60%, 60-70%, 70-80%, 80%+
    """
    if buckets is None:
        buckets = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
    results = []
    for lo, hi in buckets:
        indices = [i for i, p in enumerate(y_pred) if lo <= p < hi]
        if not indices:
            results.append({
                "bucket": f"{int(lo*100)}-{int(hi*100)}%" if hi < 1.01 else "80%+",
                "predicted_avg": (lo + min(hi, 1.0)) / 2,
                "actual_rate": None,
                "n": 0,
            })
            continue
        preds_in_bucket = [y_pred[i] for i in indices]
        trues_in_bucket = [y_true[i] for i in indices]
        results.append({
            "bucket": f"{int(lo*100)}-{int(hi*100)}%" if hi < 1.01 else "80%+",
            "predicted_avg": round(sum(preds_in_bucket) / len(preds_in_bucket), 4),
            "actual_rate": round(sum(trues_in_bucket) / len(trues_in_bucket), 4),
            "n": len(indices),
        })
    return results


def compute_model_metrics(
    fte_df,
    lr_model,
    xgb_model,
    feature_names: List[str],
    weights: Optional[dict] = None,
    lambda_decay: float = 0.05,
) -> dict:
    """
    Compute log loss, Brier score, AUC, and calibration on FTE holdout data.
    Uses most recent 2 seasons as test set.
    """
    log.info("Computing model performance metrics...")

    from ml_models import build_feature_matrix, lr_predict, xgb_predict

    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Use last 2 seasons as test
    max_season = int(fte_df["season"].max())
    test_df = fte_df[fte_df["season"] >= max_season - 1]
    train_df = fte_df[fte_df["season"] < max_season - 1]

    X_test, y_test, w_test = build_feature_matrix(test_df, {}, lambda_decay=lambda_decay)

    if len(y_test) < 10:
        log.warning("Not enough test data for metrics")
        return {"log_loss": None, "brier_score": None, "auc": None, "calibration": []}

    lr_probs = lr_predict(lr_model, X_test)
    xgb_probs = xgb_predict(xgb_model, X_test)

    # Ensemble (use equal base weights since we don't have ELO/pyth/eff for historical)
    ensemble_probs = [
        compute_ensemble_prob(lr, xgb, lr, lr, lr, {"lr": 0.4, "xgb": 0.35, "elo": 0.25/3, "pyth": 0.25/3, "eff": 0.25/3})
        for lr, xgb in zip(lr_probs, xgb_probs)
    ]

    y_true = list(y_test.astype(int))
    y_pred = ensemble_probs

    ll = calculate_log_loss(y_true, y_pred)
    bs = calculate_brier_score(y_true, y_pred)
    auc = calculate_auc(y_true, y_pred)
    cal = calibration_buckets(y_true, y_pred)

    log.info(f"Model metrics — LogLoss: {ll:.4f}, Brier: {bs:.4f}, AUC: {auc:.4f}")

    return {
        "log_loss": round(ll, 4),
        "brier_score": round(bs, 4),
        "auc": round(auc, 4),
        "calibration": cal,
        "n_test_games": len(y_true),
    }
