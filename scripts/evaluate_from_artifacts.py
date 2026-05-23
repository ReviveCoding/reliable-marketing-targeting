from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reliable_marketing.data import load_census_income, resolve_data_dir
from reliable_marketing.features import apply_rare_category_maps, build_pht_features
from reliable_marketing.metrics import calibration_bins, evaluate_binary_scores, target_attainment, weighted_confusion
from reliable_marketing.models import safe_logit

MODEL_CONFIG = {
    "practical": {
        "display_name": "REFERENCE_practical_LightGBM",
        "path": ("practical", "final_practical_model_bundle.joblib"),
    },
    "monotonic": {
        "display_name": "Monotonic_constrained_LightGBM",
        "path": ("monotonic_lgbm", "monotonic_constrained_lgbm_sigmoid.joblib"),
    },
    "logistic_classweighted": {
        "display_name": "LR_L2_classpow035_C1",
        "path": ("logistic_classweighted", "LR_L2_classpow035_C1_sigmoid_calibrated_bundle.joblib"),
    },
    "logistic_population": {
        "display_name": "LR_L2_population_C1",
        "path": ("logistic_population", "LR_L2_population_C1_sigmoid_calibrated_bundle.joblib"),
    },
}


def as_lgbm_input(xmat, model):
    """Return the encoded matrix without densifying sparse features.

    LightGBM can score the sparse matrix directly. Keeping the sparse matrix avoids
    large memory use during full-data evaluation/retraining checks. The harmless
    feature-name warning is suppressed at the prediction call sites.
    """
    return xmat


def apply_calibrator(calibrator, raw, mode: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if mode == "logit":
        return calibrator.predict_proba(safe_logit(raw).reshape(-1, 1))[:, 1]
    if mode == "raw":
        return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    raise ValueError(f"Unknown calibration mode: {mode}")


def score_bundle(model_key: str, X: pd.DataFrame, artifact_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cfg = MODEL_CONFIG[model_key]
    bundle = joblib.load(artifact_dir.joinpath(*cfg["path"]))

    if model_key == "practical":
        xmat = bundle["feature_encoder"].transform(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            raw = bundle["lightgbm_model"].predict_proba(as_lgbm_input(xmat, bundle["lightgbm_model"]))[:, 1]
        calibrated = apply_calibrator(bundle["sigmoid_calibrator"], raw, "logit")
        return raw, calibrated

    if model_key == "monotonic":
        xmat = bundle["feature_encoder"].transform(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            raw = bundle["model"].predict_proba(as_lgbm_input(xmat, bundle["model"]))[:, 1]
        calibrated = apply_calibrator(bundle["sigmoid_calibrator"], raw, "raw")
        return raw, calibrated

    if model_key in {"logistic_classweighted", "logistic_population"}:
        xt = apply_rare_category_maps(build_pht_features(X), bundle["rare_maps"])
        xmat = bundle["preprocessor"].transform(xt)
        raw = bundle["logistic_model"].predict_proba(xmat)[:, 1]
        calibrated = apply_calibrator(bundle["sigmoid_calibrator"], raw, "logit")
        return raw, calibrated

    raise ValueError(f"Unknown model key: {model_key}")


def load_split_indices(path: Path) -> dict[str, np.ndarray]:
    split = pd.read_csv(path)
    return {name: g["row_id"].to_numpy(dtype=int) for name, g in split.groupby("split")}


def build_topk_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in metrics.iterrows():
        for pct in [1, 5, 10, 20]:
            rows.append(
                {
                    "model": r["model"],
                    "top_k_percent": pct,
                    "precision_w": r.get(f"Precision@Top{pct}_w"),
                    "recall_w": r.get(f"Recall@Top{pct}_w"),
                    "capture_rate_w": r.get(f"CaptureRate@Top{pct}_w"),
                    "lift_w": r.get(f"Lift@Top{pct}_w"),
                }
            )
    return pd.DataFrame(rows)


def compare_with_stored(recomputed: pd.DataFrame, stored_path: Path) -> pd.DataFrame:
    if not stored_path.exists():
        return pd.DataFrame()
    stored = pd.read_csv(stored_path)
    common_metrics = [
        "AP_w",
        "ROC_AUC_w",
        "Precision_w",
        "Recall_w",
        "F1_w",
        "Brier_w",
        "LogLoss_w",
        "Precision@Top5_w",
        "Recall@Top5_w",
        "Lift@Top5_w",
        "Precision@Top10_w",
        "Recall@Top10_w",
        "Lift@Top10_w",
        "Precision@Top20_w",
        "Recall@Top20_w",
        "Lift@Top20_w",
    ]
    rows = []
    merged = stored.merge(recomputed, on="model", how="inner", suffixes=("_stored", "_recomputed"))
    for _, r in merged.iterrows():
        for m in common_metrics:
            s_col, rc_col = f"{m}_stored", f"{m}_recomputed"
            if s_col in merged.columns and rc_col in merged.columns:
                rows.append(
                    {
                        "model": r["model"],
                        "metric": m,
                        "stored": r[s_col],
                        "recomputed": r[rc_col],
                        "abs_delta": abs(r[s_col] - r[rc_col]),
                    }
                )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> pd.DataFrame:
    artifact_dir = Path(args.artifact_dir)
    out_metrics = ROOT / "outputs" / "metrics"
    out_tables = ROOT / "outputs" / "tables"
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(args.data_zip, args.data_dir)
    X, y, w = load_census_income(data_dir)
    splits = load_split_indices(Path(args.split_indices))
    cal_idx = splits["calibration"]
    test_idx = splits["test"]
    eval_idx = np.concatenate([cal_idx, test_idx])
    cal_pos = np.arange(len(cal_idx))
    test_pos = np.arange(len(cal_idx), len(eval_idx))
    X_eval = X.iloc[eval_idx].reset_index(drop=True)
    y_eval = y.iloc[eval_idx].reset_index(drop=True)
    w_eval = w.iloc[eval_idx].reset_index(drop=True)

    requested = list(MODEL_CONFIG) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    bad = sorted(set(requested) - set(MODEL_CONFIG))
    if bad:
        raise ValueError(f"Unknown model key(s): {bad}. Valid: {sorted(MODEL_CONFIG)}")

    rows = []
    all_bins = []
    all_confusions = []
    for key in requested:
        print(f"Recomputing held-out metrics for {key}...", flush=True)
        _, score = score_bundle(key, X_eval, artifact_dir)
        row = evaluate_binary_scores(
            MODEL_CONFIG[key]["display_name"],
            y_eval.iloc[cal_pos],
            score[cal_pos],
            w_eval.iloc[cal_pos],
            y_eval.iloc[test_pos],
            score[test_pos],
            w_eval.iloc[test_pos],
        )
        row["model_key"] = key
        rows.append(row)

        bins = calibration_bins(y_eval.iloc[test_pos], score[test_pos], sample_weight=w_eval.iloc[test_pos], n_bins=10)
        bins.insert(0, "model", MODEL_CONFIG[key]["display_name"])
        bins.insert(1, "model_key", key)
        all_bins.append(bins)

        cm = weighted_confusion(y_eval.iloc[test_pos], score[test_pos], row["threshold"], w_eval.iloc[test_pos]).reset_index(names="true_label")
        cm.insert(0, "model", MODEL_CONFIG[key]["display_name"])
        cm.insert(1, "model_key", key)
        all_confusions.append(cm)

    metrics = pd.DataFrame(rows)
    # Keep the most business-relevant columns near the front.
    front = [
        "model_key",
        "model",
        "threshold",
        "AP_w",
        "PR_AUC_trapz_w",
        "ROC_AUC_w",
        "Precision_w",
        "Recall_w",
        "F1_w",
        "Brier_w",
        "LogLoss_w",
        "ECE10_w",
        "Precision@Top1_w",
        "Lift@Top1_w",
        "Precision@Top5_w",
        "Lift@Top5_w",
        "Recall@Top5_w",
        "Precision@Top10_w",
        "Lift@Top10_w",
        "Recall@Top10_w",
        "Precision@Top20_w",
        "Lift@Top20_w",
        "Recall@Top20_w",
    ]
    metrics = metrics[[c for c in front if c in metrics.columns] + [c for c in metrics.columns if c not in front]]
    metrics.to_csv(out_metrics / "recomputed_selected_four_model_comparison.csv", index=False)
    build_topk_table(metrics).to_csv(out_tables / "recomputed_topk_policy_by_model.csv", index=False)
    pd.concat(all_bins, ignore_index=True).to_csv(out_tables / "calibration_bins_by_model.csv", index=False)
    pd.concat(all_confusions, ignore_index=True).to_csv(out_tables / "weighted_confusion_by_model.csv", index=False)

    targets = []
    for _, r in metrics.iterrows():
        targets.append(target_attainment(r).assign(model=r["model"], model_key=r["model_key"]))
    pd.concat(targets, ignore_index=True).to_csv(out_metrics / "recomputed_target_attainment.csv", index=False)

    cmp = compare_with_stored(metrics, out_metrics / "selected_four_model_comparison.csv")
    if not cmp.empty:
        cmp.to_csv(out_metrics / "recomputed_vs_stored_metric_deltas.csv", index=False)

    display_cols = ["model", "AP_w", "PR_AUC_trapz_w", "F1_w", "Brier_w", "ECE10_w", "Precision@Top5_w", "Lift@Top5_w"]
    print(metrics[display_cols].to_string(index=False))
    print(f"Saved recomputed metrics to {out_metrics / 'recomputed_selected_four_model_comparison.csv'}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Recompute held-out metrics from submitted model artifacts.")
    p.add_argument("--models", default="all", help="all or comma-separated: practical,monotonic,logistic_classweighted,logistic_population")
    p.add_argument("--data-zip", default=str(ROOT / "data" / "TakeHomeProject.zip"))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--artifact-dir", default=str(ROOT / "artifacts"))
    p.add_argument("--split-indices", default=str(ROOT / "artifacts" / "shared" / "split_indices.csv"))
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
