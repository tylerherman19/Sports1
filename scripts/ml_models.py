"""
ml_models.py
System 1: Logistic Regression with isotonic calibration.
System 9: XGBoost classifier.
Both trained on FiveThirtyEight historical NFL data.
Exports model weights as JSON-serializable dicts for browser-side inference.
"""

import logging
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

log = logging.getLogger(__name__)

FEATURE_NAMES_LR = [
    "elo_diff",
    "offensive_rating_diff",
    "defensive_rating_diff",
    "home_field",
    "rest_days_diff",
    "pythagorean_diff",
    "net_efficiency_diff",
    "turnover_diff_adjusted",
    "travel_distance_diff",
    "last5_win_rate_diff",
]

FEATURE_NAMES_XGB = FEATURE_NAMES_LR + [
    "pace_diff",
    "turnover_rate_diff",
    "third_down_diff",
    "red_zone_diff",
    "penalty_yards_diff",
    "time_of_possession_diff",
]


# ---------------------------------------------------------------------------
# Feature construction from FTE data
# ---------------------------------------------------------------------------

def build_feature_matrix(
    fte_df: pd.DataFrame,
    team_stats: Dict[str, dict],
    lambda_decay: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build training feature matrix X, labels y, and sample weights w from FTE history.
    Uses exponential decay: weight = e^(-lambda * weeks_ago)

    Returns (X, y, weights) all as numpy arrays.
    """
    log.info("Building feature matrix from FTE data...")

    rows = []
    labels = []
    weights = []

    # Track rolling ELOs as we walk through history
    from elo_model import build_elo_from_fte, DEFAULT_ELO
    from pythagorean import pythagorean_expectation

    # We need per-game features; use FTE's built-in elo columns where available
    max_date = fte_df["date"].max()

    for idx, row in fte_df.sort_values("date").iterrows():
        # Skip if missing key data
        if pd.isna(row.get("elo1_pre")) or pd.isna(row.get("elo2_pre")):
            continue

        elo_diff = float(row["elo1_pre"]) - float(row["elo2_pre"])
        home_win = 1 if float(row["score1"]) > float(row["score2"]) else 0

        # Exponential decay weight based on how many weeks ago
        days_ago = (max_date - row["date"]).days
        weeks_ago = days_ago / 7.0
        w = math.exp(-lambda_decay * weeks_ago)

        # Features from FTE columns (elo_prob, qb_value, etc.)
        off_diff = float(row.get("quality1", 0) or 0) - float(row.get("quality2", 0) or 0)
        def_diff = 0.0  # not directly available in FTE CSV; use 0

        # Pythagorean — not in FTE, estimate from elo
        pyth_diff = elo_diff / 400.0 * 0.5  # rough approximation

        feature_row = [
            elo_diff,
            off_diff,
            def_diff,
            1.0,  # all FTE home team games have home field = 1
            0.0,  # rest days diff not in FTE
            pyth_diff,
            off_diff * 0.5,  # net efficiency proxy
            0.0,  # turnover diff
            0.0,  # travel distance
            0.0,  # last5 win rate diff
        ]

        rows.append(feature_row)
        labels.append(home_win)
        weights.append(w)

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    w = np.array(weights, dtype=float)
    w = w / w.sum() * len(w)  # normalize so mean weight = 1

    log.info(f"Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, w


# ---------------------------------------------------------------------------
# Logistic Regression (System 1)
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
) -> Tuple[object, dict]:
    """
    Train logistic regression with isotonic calibration.
    Returns (calibrated_pipeline, weight_dict).
    weight_dict has 'intercept' and 'coefficients' for browser-side inference.
    """
    log.info("Training logistic regression...")

    base_lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(base_lr, method="isotonic", cv=5)
    calibrated.fit(X, y, sample_weight=sample_weights)

    # Cross-val for logging
    cv_scores = cross_val_score(
        base_lr, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_log_loss",
        fit_params={"sample_weight": sample_weights},
    )
    log.info(f"LR CV log-loss: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Extract weights from the base (uncalibrated) estimator for browser use
    # The calibrated model wraps multiple base estimators — average the coefficients
    coef_list = []
    intercept_list = []
    for estimator in calibrated.calibrated_classifiers_:
        inner = estimator.estimator
        coef_list.append(inner.coef_[0])
        intercept_list.append(inner.intercept_[0])

    avg_coef = np.mean(coef_list, axis=0)
    avg_intercept = float(np.mean(intercept_list))

    weight_dict = {
        "intercept": avg_intercept,
        "coefficients": {name: float(c) for name, c in zip(FEATURE_NAMES_LR, avg_coef)},
    }

    log.info(f"LR weights: {weight_dict['coefficients']}")
    return calibrated, weight_dict


def lr_predict(model, X: np.ndarray) -> np.ndarray:
    """Return calibrated win probabilities for home team."""
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# XGBoost (System 9)
# ---------------------------------------------------------------------------

def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    feature_names: List[str] = None,
) -> Tuple[object, dict]:
    """
    Train XGBoost with cross-validation hyperparameter tuning.
    Returns (xgb_model, info_dict).
    """
    log.info("Training XGBoost...")
    if feature_names is None:
        feature_names = FEATURE_NAMES_LR  # may be subset if extra features absent

    # Simple grid search over key hyperparameters
    best_score = float("inf")
    best_params = {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.8}

    param_grid = [
        {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 150, "subsample": 0.8},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.8},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 150, "subsample": 0.7},
    ]

    for params in param_grid:
        model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
            n_jobs=-1,
        )
        scores = cross_val_score(
            model, X, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="neg_log_loss",
            fit_params={"sample_weight": sample_weights},
        )
        mean_score = -scores.mean()
        log.info(f"XGB params {params}: log-loss {mean_score:.4f}")
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    log.info(f"Best XGB params: {best_params}, log-loss: {best_score:.4f}")
    best_model = xgb.XGBClassifier(
        **best_params,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1,
    )
    best_model.fit(X, y, sample_weight=sample_weights)

    info = {
        "best_params": best_params,
        "cv_log_loss": best_score,
        "feature_names": feature_names,
        "feature_importances": {
            name: float(imp)
            for name, imp in zip(feature_names, best_model.feature_importances_)
        },
    }

    return best_model, info


def xgb_predict(model, X: np.ndarray) -> np.ndarray:
    """Return XGBoost win probabilities for home team."""
    return model.predict_proba(X)[:, 1]


def export_xgb_game_probs(
    xgb_model,
    current_games: List[dict],
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Run XGBoost inference on current week's games and return per-game probs.
    game_key = f"{home_team}_vs_{away_team}"
    """
    probs = {}
    for game in current_games:
        home = game["home_team"]
        away = game["away_team"]
        feats = game.get("features", {})
        row = np.array([[feats.get(f, 0.0) for f in feature_names]])
        prob = float(xgb_model.predict_proba(row)[0, 1])
        key = f"{home}_vs_{away}"
        probs[key] = prob
    return probs
