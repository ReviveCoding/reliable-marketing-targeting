from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(args: argparse.Namespace) -> None:
    artifact_path = Path(args.artifact)
    bundle = joblib.load(artifact_path)
    model = bundle["lightgbm_model"]
    encoder = bundle["feature_encoder"]
    names = list(getattr(encoder, "feature_names", []))
    importances = np.asarray(model.feature_importances_, dtype=float)
    if not names or len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]
    gain_importance = None
    try:
        gain_importance = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    except Exception:
        gain_importance = importances.astype(float)
    out = pd.DataFrame({
        "feature": names,
        "split_importance": importances,
        "gain_importance": gain_importance,
    }).sort_values(["gain_importance", "split_importance"], ascending=False)

    out_table = Path(args.out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_table, index=False)

    if args.out_figure:
        try:
            import matplotlib.pyplot as plt
            top = out.head(args.top_n).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, 5.5))
            ax.barh(top["feature"], top["gain_importance"])
            ax.set_xlabel("LightGBM gain importance")
            ax.set_title(f"Practical LightGBM top {len(top)} features")
            fig.tight_layout()
            out_fig = Path(args.out_figure)
            out_fig.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_fig, dpi=160)
            plt.close(fig)
        except Exception as exc:
            print(f"Skipped feature-importance figure: {exc}")
    print(f"Saved practical feature importance to {out_table}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Practical LightGBM feature-importance table and plot from the saved artifact.")
    p.add_argument("--artifact", default=str(ROOT / "artifacts" / "practical" / "final_practical_model_bundle.joblib"))
    p.add_argument("--out-table", default=str(ROOT / "outputs" / "tables" / "practical_lgbm_feature_importance.csv"))
    p.add_argument("--out-figure", default=str(ROOT / "outputs" / "figures" / "practical_lgbm_feature_importance_top20.png"))
    p.add_argument("--top-n", type=int, default=20)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
