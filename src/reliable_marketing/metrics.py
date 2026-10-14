from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true, score, sample_weight=None) -> float:
    """Choose the observed score threshold with the highest PR-curve F1."""
    precision, recall, thresholds = precision_recall_curve(
        y_true, score, sample_weight=sample_weight
    )
    if len(thresholds) == 0:
        return 0.5
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    return float(thresholds[int(np.nanargmax(f1))])


def pr_auc_trapezoid(y_true, score, sample_weight=None) -> float:
    """Compute trapezoidal area under the precision-recall curve.

    Average Precision (AP) remains the primary PR-curve summary because it is
    less optimistic than simple interpolation for step-wise PR curves. This
    supplementary metric is included because some reviewers use PR-AUC wording.
    """
    precision, recall, _ = precision_recall_curve(y_true, score, sample_weight=sample_weight)
    order = np.argsort(recall)
    return float(auc(recall[order], precision[order]))


def calibration_bins(y_true, score, sample_weight=None, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    s = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    w = np.ones_like(s, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Include 1.0 in the final bin.
    bin_id = np.minimum(np.searchsorted(edges, s, side="right") - 1, n_bins - 1)
    bin_id = np.maximum(bin_id, 0)
    rows = []
    total_w = w.sum()
    for b in range(n_bins):
        mask = bin_id == b
        if not np.any(mask):
            rows.append(
                {
                    "bin": b,
                    "score_min": edges[b],
                    "score_max": edges[b + 1],
                    "n": 0,
                    "weight_share": 0.0,
                    "avg_predicted_score": np.nan,
                    "observed_positive_rate": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        bw = w[mask].sum()
        avg_pred = float(np.average(s[mask], weights=w[mask]))
        obs = float(np.average(y[mask], weights=w[mask]))
        rows.append(
            {
                "bin": b,
                "score_min": edges[b],
                "score_max": edges[b + 1],
                "n": int(mask.sum()),
                "weight_share": float(bw / total_w) if total_w else 0.0,
                "avg_predicted_score": avg_pred,
                "observed_positive_rate": obs,
                "abs_gap": abs(avg_pred - obs),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, score, sample_weight=None, n_bins: int = 10) -> float:
    bins = calibration_bins(y_true, score, sample_weight=sample_weight, n_bins=n_bins)
    valid = bins["n"] > 0
    return float((bins.loc[valid, "weight_share"] * bins.loc[valid, "abs_gap"]).sum())


def topk_metrics(y_true, score, sample_weight, pct: float):
    y = np.asarray(y_true)
    s = np.asarray(score)
    w = np.asarray(sample_weight, dtype=float)
    k = max(1, int(np.ceil(len(y) * pct)))
    idx = np.argsort(-s)[:k]
    tp_w = (w[idx] * y[idx]).sum()
    pred_w = w[idx].sum()
    pos_w = (w * y).sum()
    base_rate = pos_w / w.sum()
    precision = tp_w / pred_w if pred_w > 0 else 0.0
    recall = tp_w / pos_w if pos_w > 0 else 0.0
    lift = precision / base_rate if base_rate > 0 else 0.0
    return precision, recall, lift


def evaluate_binary_scores(name, y_cal, p_cal, w_cal, y_test, p_test, w_test):
    threshold = best_f1_threshold(y_cal, p_cal, sample_weight=w_cal)
    pred = (p_test >= threshold).astype(int)
    clipped = np.clip(p_test, 1e-6, 1 - 1e-6)
    row = {
        "model": name,
        "threshold": threshold,
        "AP_w": average_precision_score(y_test, p_test, sample_weight=w_test),
        "PR_AUC_trapz_w": pr_auc_trapezoid(y_test, p_test, sample_weight=w_test),
        "ROC_AUC_w": roc_auc_score(y_test, p_test, sample_weight=w_test),
        "Precision_w": precision_score(y_test, pred, sample_weight=w_test, zero_division=0),
        "Recall_w": recall_score(y_test, pred, sample_weight=w_test, zero_division=0),
        "F1_w": f1_score(y_test, pred, sample_weight=w_test, zero_division=0),
        "Brier_w": brier_score_loss(y_test, np.clip(p_test, 0, 1), sample_weight=w_test),
        "LogLoss_w": log_loss(y_test, clipped, sample_weight=w_test),
        "ECE10_w": expected_calibration_error(y_test, p_test, sample_weight=w_test, n_bins=10),
    }
    for pct in [0.01, 0.05, 0.10, 0.20]:
        precision, recall, lift = topk_metrics(y_test, p_test, w_test, pct)
        label = int(pct * 100)
        row[f"Precision@Top{label}_w"] = precision
        row[f"Recall@Top{label}_w"] = recall
        row[f"Lift@Top{label}_w"] = lift
        row[f"CaptureRate@Top{label}_w"] = recall
    return row


def target_attainment(metrics_row: dict) -> pd.DataFrame:
    targets = {
        "AP_w": (">=", 0.70),
        "PR_AUC_trapz_w": (">=", 0.70),
        "Precision_w": (">=", 0.65),
        "Recall_w": (">=", 0.65),
        "F1_w": (">=", 0.65),
        "Precision@Top5_w": (">=", 0.70),
        "Recall@Top5_w": (">=", 0.55),
        "Lift@Top5_w": (">=", 11.0),
        "ROC_AUC_w": (">=", 0.95),
        "Brier_w": ("<=", 0.04),
        "LogLoss_w": ("<=", 0.12),
        "ECE10_w": ("<=", 0.03),
        "Precision@Top10_w": (">=", 0.45),
        "Recall@Top10_w": (">=", 0.75),
        "Lift@Top10_w": (">=", 7.0),
    }
    rows = []
    for metric, (direction, target) in targets.items():
        value = metrics_row.get(metric)
        if value is None or pd.isna(value):
            hit = False
        else:
            hit = value <= target if direction == "<=" else value >= target
        rows.append({"metric": metric, "target": target, "direction": direction, "value": value, "hit": bool(hit)})
    return pd.DataFrame(rows)


def weighted_confusion(y_true, score, threshold, sample_weight):
    pred = (np.asarray(score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, sample_weight=sample_weight, labels=[0, 1])
    return pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
