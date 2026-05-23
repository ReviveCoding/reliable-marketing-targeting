from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


def positive_class_power_weights(y, sample_weight, class_weight_power: float = 0.0):
    """Combine population/sample weights with optional positive-class emphasis."""
    y_arr = np.asarray(y)
    w = np.asarray(sample_weight, dtype=float)
    if class_weight_power <= 0:
        return w
    pos = y_arr.sum()
    neg = len(y_arr) - pos
    pos_weight = neg / max(pos, 1)
    return w * np.where(y_arr == 1, pos_weight ** class_weight_power, 1.0)


def safe_logit(p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_sigmoid_calibrator(score, y, sample_weight=None, max_iter: int = 300) -> LogisticRegression:
    """Fit a Platt-style sigmoid calibrator on one-dimensional scores."""
    cal = LogisticRegression(max_iter=max_iter)
    cal.fit(safe_logit(score).reshape(-1, 1), y, sample_weight=sample_weight)
    return cal


def apply_sigmoid_calibrator(calibrator: LogisticRegression, score):
    return calibrator.predict_proba(safe_logit(score).reshape(-1, 1))[:, 1]


def practical_lgbm_params(n_estimators: int | None = None, quick: bool = False) -> dict:
    params = dict(
        objective="binary",
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=50,
        max_depth=10,
        min_child_samples=200,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.5,
        min_split_gain=0.0,
        random_state=55,
        n_jobs=4,
        verbose=-1,
        force_col_wise=True,
        max_bin=128,
    )
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    if quick:
        params["min_child_samples"] = min(params["min_child_samples"], 20)
        params["num_leaves"] = min(params["num_leaves"], 20)
    return params


def monotonic_lgbm_params(monotone_constraints: list[int], n_estimators: int | None = None, quick: bool = False) -> dict:
    params = dict(
        objective="binary",
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=40,
        max_depth=8,
        min_child_samples=250,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=2.0,
        min_split_gain=0.0,
        random_state=42,
        n_jobs=4,
        verbose=-1,
        force_col_wise=True,
        max_bin=128,
        monotone_constraints=monotone_constraints,
    )
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    if quick:
        params["min_child_samples"] = min(params["min_child_samples"], 20)
        params["num_leaves"] = min(params["num_leaves"], 20)
    return params


def make_lgbm_classifier(params: dict) -> LGBMClassifier:
    return LGBMClassifier(**params)
