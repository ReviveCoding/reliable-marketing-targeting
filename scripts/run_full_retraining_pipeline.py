from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reliable_marketing.data import load_census_income, resolve_data_dir
from reliable_marketing.features import (
    FeatureEncoder,
    apply_rare_category_maps,
    build_pht_features,
    fit_rare_category_maps,
)
from reliable_marketing.metrics import evaluate_binary_scores, target_attainment
from reliable_marketing.models import (
    apply_sigmoid_calibrator,
    fit_sigmoid_calibrator,
    monotonic_lgbm_params,
    positive_class_power_weights,
    practical_lgbm_params,
)

MODEL_CHOICES = ["practical", "monotonic", "logistic_classweighted", "logistic_population"]


def make_unique_dir(path: Path, overwrite: bool = False) -> Path:
    """Create a run directory without overwriting previous training outputs.

    Final submitted model artifacts live under ROOT/artifacts and are never used as
    a retraining output location. If the requested path already exists, this
    function appends _001, _002, ... unless overwrite=True is explicitly used.
    """
    path = path.resolve()
    if overwrite:
        path.mkdir(parents=True, exist_ok=True)
        return path
    candidate = path
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{suffix:03d}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def default_run_name(models_arg: str, quick_rows: int | None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_part = "all" if models_arg == "all" else models_arg.replace(",", "_").replace(" ", "")
    quick_part = f"quick{quick_rows}" if quick_rows is not None else "full"
    return f"run_{stamp}_{quick_part}_{model_part}"


def as_lgbm_input(xmat, model):
    """Return a feature-named DataFrame when the fitted LightGBM model expects feature names."""
    names = getattr(model, "feature_name_", None)
    if names is None:
        return xmat
    if hasattr(xmat, "toarray"):
        xmat = xmat.toarray()
    return pd.DataFrame(xmat, columns=list(names))


def split_data(X, y, w, quick_rows: int | None = None):
    if quick_rows is not None:
        X = X.iloc[:quick_rows].reset_index(drop=True)
        y = y.iloc[:quick_rows].reset_index(drop=True)
        w = w.iloc[:quick_rows].reset_index(drop=True)

    idx = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=42, stratify=y)
    train_idx, cal_idx = train_test_split(
        train_val_idx, test_size=0.25, random_state=42, stratify=np.asarray(y)[train_val_idx]
    )
    return X, y, w, train_idx, cal_idx, test_idx


def save_split_indices(out_dir: Path, train_idx, cal_idx, test_idx):
    rows = []
    for name, values in [("train", train_idx), ("calibration", cal_idx), ("test", test_idx)]:
        rows.extend({"row_id": int(i), "split": name} for i in values)
    pd.DataFrame(rows).sort_values("row_id").to_csv(out_dir / "split_indices.csv", index=False)


def calibration_and_metrics(model_name, raw_cal, raw_test, y_cal, y_test, w_cal, w_test):
    calibrator = fit_sigmoid_calibrator(raw_cal, y_cal, sample_weight=w_cal)
    p_cal = apply_sigmoid_calibrator(calibrator, raw_cal)
    p_test = apply_sigmoid_calibrator(calibrator, raw_test)
    row = evaluate_binary_scores(model_name, y_cal, p_cal, w_cal, y_test, p_test, w_test)
    return calibrator, p_cal, p_test, row


def feature_constraints_from_map(feature_names, constraint_map_path: Path) -> list[int]:
    if constraint_map_path.exists():
        df = pd.read_csv(constraint_map_path)
        mapping = dict(zip(df["feature"].astype(str), df["monotone_constraint"].astype(int)))
        return [int(mapping.get(str(name), 0)) for name in feature_names]
    positive = {
        "weeks worked in year",
        "capital gains",
        "dividends from stocks",
        "any_investment_signal",
        "employment_intensity",
        "num persons worked for employer",
        "log1p_capital gains",
        "log1p_dividends from stocks",
    }
    return [1 if str(name) in positive else 0 for name in feature_names]


def train_lgbm_route(model_key, X, y, w, train_idx, cal_idx, test_idx, out_dir: Path, n_estimators: int | None, quick: bool):
    feature_encoder = FeatureEncoder.fit(X.iloc[train_idx], min_count=150 if not quick else 5, feature_version="pht")
    X_train = feature_encoder.transform(X.iloc[train_idx])
    X_cal = feature_encoder.transform(X.iloc[cal_idx])
    X_test = feature_encoder.transform(X.iloc[test_idx])

    if model_key == "monotonic":
        constraints = feature_constraints_from_map(
            feature_encoder.feature_names,
            ROOT / "outputs" / "tables" / "monotonic_constraint_map.csv",
        )
        params = monotonic_lgbm_params(constraints, n_estimators=n_estimators, quick=quick)
        model_name = "Monotonic_constrained_LightGBM__retrained"
    else:
        params = practical_lgbm_params(n_estimators=n_estimators, quick=quick)
        model_name = "REFERENCE_practical_LightGBM__retrained"

    model = LGBMClassifier(**params)
    train_weights = positive_class_power_weights(y.iloc[train_idx], w.iloc[train_idx], class_weight_power=0.35)
    model.fit(X_train, y.iloc[train_idx], sample_weight=train_weights)

    raw_cal = model.predict_proba(as_lgbm_input(X_cal, model))[:, 1]
    raw_test = model.predict_proba(as_lgbm_input(X_test, model))[:, 1]
    calibrator, p_cal, p_test, row = calibration_and_metrics(
        model_name, raw_cal, raw_test, y.iloc[cal_idx], y.iloc[test_idx], w.iloc[cal_idx], w.iloc[test_idx]
    )

    bundle = {
        "model_key": model_key,
        "model_name": model_name,
        "feature_encoder": feature_encoder,
        "model": model,
        "sigmoid_calibrator": calibrator,
        "class_weight_power": 0.35,
        "params": params,
        "metrics": row,
    }
    model_dir = out_dir / "artifacts" / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / f"{model_key}_retrained_bundle.joblib")
    if hasattr(model, "booster_"):
        model.booster_.save_model(str(model_dir / f"{model_key}_retrained_lightgbm_booster.txt"))
    return row


def make_logistic_preprocessor(X_train_fe: pd.DataFrame):
    num_cols = X_train_fe.select_dtypes(exclude="object").columns.tolist()
    cat_cols = X_train_fe.select_dtypes(include="object").columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))]), num_cols),
            (
                "cat",
                Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")), ("oh", OneHotEncoder(handle_unknown="ignore"))]),
                cat_cols,
            ),
        ],
        sparse_threshold=1.0,
    )
    return preprocessor, num_cols, cat_cols


def train_logistic_route(model_key, X, y, w, train_idx, cal_idx, test_idx, out_dir: Path, quick: bool):
    X_train_fe = build_pht_features(X.iloc[train_idx])
    rare_maps = fit_rare_category_maps(X_train_fe, min_count=150 if not quick else 5)
    X_train_fe = apply_rare_category_maps(X_train_fe, rare_maps)
    X_cal_fe = apply_rare_category_maps(build_pht_features(X.iloc[cal_idx]), rare_maps)
    X_test_fe = apply_rare_category_maps(build_pht_features(X.iloc[test_idx]), rare_maps)

    preprocessor, num_cols, cat_cols = make_logistic_preprocessor(X_train_fe)
    X_train_mat = preprocessor.fit_transform(X_train_fe)
    X_cal_mat = preprocessor.transform(X_cal_fe)
    X_test_mat = preprocessor.transform(X_test_fe)

    class_power = 0.35 if model_key == "logistic_classweighted" else 0.0
    model_name = "LR_L2_classpow035_C1__retrained" if class_power else "LR_L2_population_C1__retrained"
    train_weights = positive_class_power_weights(y.iloc[train_idx], w.iloc[train_idx], class_weight_power=class_power)
    logistic = LogisticRegression(C=1.0, max_iter=160 if not quick else 80, solver="lbfgs")
    logistic.fit(X_train_mat, y.iloc[train_idx], sample_weight=train_weights)

    raw_cal = logistic.predict_proba(X_cal_mat)[:, 1]
    raw_test = logistic.predict_proba(X_test_mat)[:, 1]
    calibrator, p_cal, p_test, row = calibration_and_metrics(
        model_name, raw_cal, raw_test, y.iloc[cal_idx], y.iloc[test_idx], w.iloc[cal_idx], w.iloc[test_idx]
    )

    bundle = {
        "model_key": model_key,
        "model_name": model_name,
        "preprocessor": preprocessor,
        "rare_maps": rare_maps,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "logistic_model": logistic,
        "sigmoid_calibrator": calibrator,
        "class_weight_power": class_power,
        "C": 1.0,
        "metrics": row,
    }
    model_dir = out_dir / "artifacts" / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / f"{model_key}_retrained_bundle.joblib")
    return row


def run(args):
    if args.out_dir:
        requested_out_dir = Path(args.out_dir)
    else:
        run_name = args.run_name or default_run_name(args.models, args.quick_rows)
        requested_out_dir = Path(args.base_out_dir) / run_name

    final_artifact_root = (ROOT / "artifacts").resolve()
    requested_resolved = requested_out_dir.resolve()
    if requested_resolved == final_artifact_root or final_artifact_root in requested_resolved.parents:
        raise ValueError(
            "Refusing to write retraining outputs inside the submitted artifacts/ directory. "
            "Use --base-out-dir retraining_runs or --out-dir outside artifacts/."
        )

    out_dir = make_unique_dir(requested_out_dir, overwrite=args.overwrite_output)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(args.data_zip, args.data_dir)
    X, y, w = load_census_income(data_dir)
    X, y, w, train_idx, cal_idx, test_idx = split_data(X, y, w, quick_rows=args.quick_rows)
    save_split_indices(out_dir / "metrics", train_idx, cal_idx, test_idx)

    requested = MODEL_CHOICES if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    bad = sorted(set(requested) - set(MODEL_CHOICES))
    if bad:
        raise ValueError(f"Unknown model keys: {bad}. Valid choices: {MODEL_CHOICES}")

    rows = []
    quick = args.quick_rows is not None
    for key in requested:
        print(f"Retraining {key}...", flush=True)
        if key in {"practical", "monotonic"}:
            row = train_lgbm_route(key, X, y, w, train_idx, cal_idx, test_idx, out_dir, args.n_estimators, quick)
        else:
            row = train_logistic_route(key, X, y, w, train_idx, cal_idx, test_idx, out_dir, quick)
        row["model_key"] = key
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics" / "retraining_model_comparison.csv", index=False)
    if len(metrics):
        targets = []
        for _, r in metrics.iterrows():
            t = target_attainment(r).assign(model=r["model"], model_key=r["model_key"])
            targets.append(t)
        pd.concat(targets, ignore_index=True).to_csv(out_dir / "metrics" / "retraining_target_attainment.csv", index=False)

    summary = {
        "rows_used": int(len(X)),
        "quick_rows": args.quick_rows,
        "models": requested,
        "train_rows": int(len(train_idx)),
        "calibration_rows": int(len(cal_idx)),
        "test_rows": int(len(test_idx)),
        "out_dir": str(out_dir),
        "final_artifact_root_preserved": str(final_artifact_root),
        "overwrite_output": bool(args.overwrite_output),
        "note": "This script retrains selected model families from raw data into a separate run directory. Submitted final artifacts under artifacts/ are never overwritten.",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics[["model_key", "model", "AP_w", "F1_w", "Brier_w", "Precision@Top5_w", "Lift@Top5_w"]].to_string(index=False))
    print(f"Saved retraining outputs to {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(description="Retrain selected marketing models from raw census data.")
    p.add_argument("--models", default="all", help="all or comma-separated model keys: practical,monotonic,logistic_classweighted,logistic_population")
    p.add_argument("--data-zip", default=str(ROOT / "data" / "TakeHomeProject.zip"))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--base-out-dir", default=str(ROOT / "retraining_runs"), help="Base directory for timestamped retraining runs. Used when --out-dir is not provided.")
    p.add_argument("--run-name", default=None, help="Optional run folder name under --base-out-dir. Defaults to a timestamped run name.")
    p.add_argument("--out-dir", default=None, help="Optional exact output directory. If it already exists, a numeric suffix is added unless --overwrite-output is set.")
    p.add_argument("--overwrite-output", action="store_true", help="Allow writing into an existing retraining output directory. Never allowed inside artifacts/.")
    p.add_argument("--quick-rows", type=int, default=None, help="Use first N rows for fast CI/smoke retraining.")
    p.add_argument("--n-estimators", type=int, default=None, help="Override LightGBM n_estimators for fast smoke tests.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
