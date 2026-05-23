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
from reliable_marketing.segmentation import add_score_tier


def as_lgbm_input(xmat, model):
    """Return the encoded matrix without densifying sparse features.

    LightGBM can score the sparse matrix directly. Keeping the sparse matrix avoids
    large memory use during full-data evaluation/retraining checks. The harmless
    feature-name warning is suppressed at the prediction call sites.
    """
    return xmat


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_scores(model_key: str, X: pd.DataFrame, artifact_dir: Path) -> np.ndarray:
    if model_key == "practical":
        bundle = joblib.load(artifact_dir / "practical" / "final_practical_model_bundle.joblib")
        Xmat = bundle["feature_encoder"].transform(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            raw = bundle["lightgbm_model"].predict_proba(as_lgbm_input(Xmat, bundle["lightgbm_model"]))[:, 1]
        return bundle["sigmoid_calibrator"].predict_proba(safe_logit(raw).reshape(-1, 1))[:, 1]

    if model_key == "monotonic":
        bundle = joblib.load(artifact_dir / "monotonic_lgbm" / "monotonic_constrained_lgbm_sigmoid.joblib")
        Xmat = bundle["feature_encoder"].transform(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            raw = bundle["model"].predict_proba(as_lgbm_input(Xmat, bundle["model"]))[:, 1]
        return bundle["sigmoid_calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]

    logistic_artifact = {
        "logistic_classweighted": ("logistic_classweighted", "LR_L2_classpow035_C1_sigmoid_calibrated_bundle.joblib"),
        "logistic_population": ("logistic_population", "LR_L2_population_C1_sigmoid_calibrated_bundle.joblib"),
    }[model_key]
    bundle = joblib.load(artifact_dir.joinpath(*logistic_artifact))
    Xt = apply_rare_category_maps(build_pht_features(X), bundle["rare_maps"])
    Xmat = bundle["preprocessor"].transform(Xt)
    raw = bundle["logistic_model"].predict_proba(Xmat)[:, 1]
    return bundle["sigmoid_calibrator"].predict_proba(safe_logit(raw).reshape(-1, 1))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the full dataset with one of the selected four models.")
    parser.add_argument(
        "--model",
        choices=["practical", "monotonic", "logistic_classweighted", "logistic_population"],
        default="practical",
        help="Model route to use. Default: practical.",
    )
    parser.add_argument("--data-zip", default=str(ROOT / "data" / "TakeHomeProject.zip"))
    parser.add_argument("--artifact-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--include-labels", action="store_true", help="Include y_true and sample_weight in output.")
    parser.add_argument("--out", default=None, help="Output CSV path.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_zip)
    X, y, w = load_census_income(data_dir)
    score = load_scores(args.model, X, Path(args.artifact_dir))

    out = pd.DataFrame({"row_id": np.arange(len(X)), "score": score, "score_tier": add_score_tier(score)})
    if args.include_labels:
        out["y_true"] = np.asarray(y)
        out["sample_weight"] = np.asarray(w, dtype=float)

    out_path = Path(args.out) if args.out else ROOT / "outputs" / "predictions" / f"{args.model}_full_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved {out_path} {out.shape}")


if __name__ == "__main__":
    main()
