from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_from_artifacts import score_bundle
from reliable_marketing.data import load_census_income, resolve_data_dir
from reliable_marketing.features import ADV_DEGREES, PROFESSIONAL_OCC_SUBSTRINGS, build_pht_features
from reliable_marketing.segmentation import TIER_ACTIONS, TIER_LABELS, add_score_tier, score_tier_summary

PROFILE_CATEGORY_COLS = [
    "education",
    "major occupation code",
    "major industry code",
    "marital stat",
    "tax filer stat",
    "class of worker",
]


def load_splits(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.average(values, weights=weights)) if weights.sum() else float("nan")


def top_weighted_category(df: pd.DataFrame, col: str, weight_col: str = "weight") -> tuple[str, float]:
    if col not in df.columns or df.empty:
        return "", 0.0
    tmp = df.groupby(col, dropna=False)[weight_col].sum().sort_values(ascending=False)
    if tmp.empty:
        return "", 0.0
    return str(tmp.index[0]), float(tmp.iloc[0] / tmp.sum()) if tmp.sum() else 0.0


def build_profile_frame(X: pd.DataFrame, y: pd.Series, w: pd.Series, score: np.ndarray) -> pd.DataFrame:
    fe = build_pht_features(X)
    out = X.copy()
    out["y"] = np.asarray(y, dtype=int)
    out["weight"] = np.asarray(w, dtype=float)
    out["score"] = np.asarray(score, dtype=float)
    out["segment"] = add_score_tier(score).to_numpy()
    out["college_plus"] = out["education"].isin(ADV_DEGREES).astype(int)
    out["full_year_worker"] = (out["weeks worked in year"] >= 48).astype(int)
    occ = out["major occupation code"].astype(str)
    out["professional_occ_flag"] = occ.apply(lambda s: any(x in s for x in PROFESSIONAL_OCC_SUBSTRINGS)).astype(int)
    out["investment_signal"] = (
        (out["capital gains"] > 0) | (out["capital losses"] > 0) | (out["dividends from stocks"] > 0)
    ).astype(int)
    out["employment_intensity"] = out["weeks worked in year"] / 52.0
    out["income_readiness_score"] = fe.get("income_readiness_score_v2", 0.0)
    return out


def segment_profile_summary(profile: pd.DataFrame) -> pd.DataFrame:
    base_rate = weighted_mean(profile["y"], profile["weight"])
    total_w = profile["weight"].sum()
    rows = []
    for seg in TIER_LABELS:
        g = profile[profile["segment"] == seg]
        if g.empty:
            continue
        top_edu, top_edu_share = top_weighted_category(g, "education")
        top_occ, top_occ_share = top_weighted_category(g, "major occupation code")
        top_ind, top_ind_share = top_weighted_category(g, "major industry code")
        pos_rate = weighted_mean(g["y"], g["weight"])
        rows.append(
            {
                "segment": seg,
                "n": int(len(g)),
                "weighted_population_share": float(g["weight"].sum() / total_w),
                "avg_score": weighted_mean(g["score"], g["weight"]),
                "weighted_positive_rate": pos_rate,
                "lift_vs_overall": pos_rate / base_rate if base_rate else 0.0,
                "avg_age": weighted_mean(g["age"], g["weight"]),
                "avg_weeks_worked": weighted_mean(g["weeks worked in year"], g["weight"]),
                "employment_intensity": weighted_mean(g["employment_intensity"], g["weight"]),
                "full_year_worker_share": weighted_mean(g["full_year_worker"], g["weight"]),
                "college_plus_share": weighted_mean(g["college_plus"], g["weight"]),
                "professional_occ_share": weighted_mean(g["professional_occ_flag"], g["weight"]),
                "investment_signal_share": weighted_mean(g["investment_signal"], g["weight"]),
                "top_education": top_edu,
                "top_education_weight_share": top_edu_share,
                "top_occupation": top_occ,
                "top_occupation_weight_share": top_occ_share,
                "top_industry": top_ind,
                "top_industry_weight_share": top_ind_share,
                "recommended_action": TIER_ACTIONS.get(seg, "Review before activation."),
            }
        )
    return pd.DataFrame(rows)


def segment_top_categories(profile: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    rows = []
    for seg in TIER_LABELS:
        g = profile[profile["segment"] == seg]
        if g.empty:
            continue
        for col in PROFILE_CATEGORY_COLS:
            vc = g.groupby(col, dropna=False)["weight"].sum().sort_values(ascending=False).head(top_n)
            denom = vc.sum() if vc.sum() else 1.0
            for rank, (cat, ww) in enumerate(vc.items(), start=1):
                rows.append(
                    {
                        "segment": seg,
                        "feature": col,
                        "rank": rank,
                        "category": str(cat),
                        "weighted_count": float(ww),
                        "share_among_top_categories": float(ww / denom),
                    }
                )
    return pd.DataFrame(rows)


def maybe_make_segment_figures(summary: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipped segmentation figures")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(summary))
    ax.bar(x, summary["weighted_positive_rate"])
    ax.set_xticks(x)
    ax.set_xticklabels(summary["segment"], rotation=25, ha="right")
    ax.set_ylabel("Weighted $50K+ rate")
    ax.set_title("Held-out positive rate by marketing segment")
    fig.tight_layout()
    fig.savefig(out_dir / "segment_positive_rate_lift.png", dpi=160)
    plt.close(fig)

    heat_cols = [
        "avg_score",
        "weighted_positive_rate",
        "lift_vs_overall",
        "full_year_worker_share",
        "college_plus_share",
        "professional_occ_share",
        "investment_signal_share",
    ]
    mat = summary[heat_cols].astype(float)
    norm = (mat - mat.min()) / (mat.max() - mat.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(norm.to_numpy(), aspect="auto")
    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels(summary["segment"])
    ax.set_xticks(np.arange(len(heat_cols)))
    ax.set_xticklabels(heat_cols, rotation=30, ha="right")
    ax.set_title("Segment profile heatmap, normalized by column")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "segment_profile_heatmap.png", dpi=160)
    plt.close(fig)


def build_cluster_features(X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    out["age"] = X["age"]
    out["weeks worked in year"] = X["weeks worked in year"]
    out["employment_intensity"] = X["weeks worked in year"] / 52.0
    out["full_year_worker"] = (X["weeks worked in year"] >= 48).astype(int)
    out["college_plus"] = X["education"].isin(ADV_DEGREES).astype(int)
    out["professional_occ_flag"] = X["major occupation code"].astype(str).apply(lambda s: any(x in s for x in PROFESSIONAL_OCC_SUBSTRINGS)).astype(int)
    out["self_employed_flag"] = (X["own business or self employed"] > 0).astype(int)
    out["investment_signal"] = ((X["capital gains"] > 0) | (X["capital losses"] > 0) | (X["dividends from stocks"] > 0)).astype(int)
    out["log1p_capital_gains"] = np.log1p(X["capital gains"].clip(lower=0))
    out["log1p_dividends"] = np.log1p(X["dividends from stocks"].clip(lower=0))
    out["education"] = X["education"].astype(str)
    out["major occupation code"] = X["major occupation code"].astype(str)
    out["marital stat"] = X["marital stat"].astype(str)
    out["tax filer stat"] = X["tax filer stat"].astype(str)
    return out


def optional_cluster_segmentation(profile: pd.DataFrame, X: pd.DataFrame, k_values: list[int], random_state: int, max_fit_rows: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = build_cluster_features(X)
    num_cols = feats.select_dtypes(exclude="object").columns.tolist()
    cat_cols = feats.select_dtypes(include="object").columns.tolist()
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), cat_cols),
        ]
    )
    mat = pre.fit_transform(feats)
    rng = np.random.default_rng(random_state)
    if mat.shape[0] > max_fit_rows:
        fit_idx = np.sort(rng.choice(mat.shape[0], size=max_fit_rows, replace=False))
    else:
        fit_idx = np.arange(mat.shape[0])
    mat_fit = mat[fit_idx]

    scores = []
    best_model = None
    best_k = None
    best_inertia = np.inf
    for k in k_values:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=3,
            batch_size=512,
            max_iter=30,
            max_no_improvement=5,
        )
        km.fit(mat_fit)
        inertia = float(km.inertia_)
        labels_fit = km.predict(mat_fit)
        try:
            if len(np.unique(labels_fit)) > 1 and mat_fit.shape[0] > k:
                sil_rows = min(1000, mat_fit.shape[0])
                sil = float(silhouette_score(mat_fit, labels_fit, sample_size=sil_rows, random_state=random_state))
            else:
                sil = float("nan")
        except Exception:
            sil = float("nan")
        scores.append({
            "k": k,
            "selection_metric": "lowest_sample_inertia_with_silhouette_diagnostic",
            "inertia_fit_sample": inertia,
            "silhouette_fit_sample": sil,
            "fit_rows": int(len(fit_idx)),
        })
        if inertia < best_inertia:
            best_inertia = inertia
            best_model = km
            best_k = k
    if best_model is None:
        best_k = k_values[0]
        best_model = MiniBatchKMeans(n_clusters=best_k, random_state=random_state, n_init=3, batch_size=512, max_iter=30).fit(mat_fit)

    # The clustering appendix is intentionally sampled for speed and stability;
    # it is a profile-discovery check, not the activation layer used for scoring.
    labels = best_model.predict(mat_fit)
    tmp = profile.iloc[fit_idx].copy().reset_index(drop=True)
    tmp["cluster"] = labels
    rows = []
    base_rate = weighted_mean(tmp["y"], tmp["weight"])
    total_w = tmp["weight"].sum()
    for cluster, g in tmp.groupby("cluster"):
        top_edu, _ = top_weighted_category(g, "education")
        top_occ, _ = top_weighted_category(g, "major occupation code")
        pos = weighted_mean(g["y"], g["weight"])
        rows.append(
            {
                "cluster": int(cluster),
                "selected_k": int(best_k),
                "n": int(len(g)),
                "weighted_population_share": float(g["weight"].sum() / total_w),
                "weighted_positive_rate": pos,
                "lift_vs_overall": pos / base_rate if base_rate else 0.0,
                "avg_age": weighted_mean(g["age"], g["weight"]),
                "full_year_worker_share": weighted_mean(g["full_year_worker"], g["weight"]),
                "college_plus_share": weighted_mean(g["college_plus"], g["weight"]),
                "professional_occ_share": weighted_mean(g["professional_occ_flag"], g["weight"]),
                "investment_signal_share": weighted_mean(g["investment_signal"], g["weight"]),
                "top_education": top_edu,
                "top_occupation": top_occ,
            }
        )
    return pd.DataFrame(scores), pd.DataFrame(rows).sort_values("cluster")


def write_segmentation_appendix(summary: pd.DataFrame, cluster_summary: pd.DataFrame | None, out_path: Path, split_name: str) -> None:
    lines = [
        "# Segmentation Appendix",
        "",
        "The main segmentation is a score-tier profile segmentation. It converts calibrated model scores into actionable marketing tiers and then profiles each tier using demographic, employment, education, occupation, and financial-signal variables.",
        "",
        f"The performance/profile rates below are generated on the `{split_name}` split to avoid overstating segment quality from training rows.",
        "",
        "## Score-tier profile segmentation",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Optional independent clustering segmentation",
        "",
    ]
    if cluster_summary is None or cluster_summary.empty:
        lines.append("The optional clustering appendix was not generated in this run. Use `--include-clusters` to generate it.")
    else:
        lines.extend(
            [
                "The optional clustering segmentation excludes the target label and classifier score. It is provided as a profile-discovery appendix, while the score-tier segmentation remains the recommended activation layer.",
                "",
                cluster_summary.to_markdown(index=False, floatfmt=".4f"),
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    data_dir = resolve_data_dir(args.data_zip, args.data_dir)
    X, y, w = load_census_income(data_dir)

    split_name = args.split
    if split_name != "full":
        split_df = load_splits(Path(args.split_indices))
        ids = split_df.loc[split_df["split"].eq(split_name), "row_id"].to_numpy(dtype=int)
        X_use = X.iloc[ids].reset_index(drop=True)
        y_use = y.iloc[ids].reset_index(drop=True)
        w_use = w.iloc[ids].reset_index(drop=True)
    else:
        X_use = X.reset_index(drop=True)
        y_use = y.reset_index(drop=True)
        w_use = w.reset_index(drop=True)

    _, score_use = score_bundle(args.model, X_use, Path(args.artifact_dir))

    profile = build_profile_frame(X_use, y_use, w_use, score_use)
    summary = segment_profile_summary(profile)
    # Keep classic score-tier summary too, for backward compatibility with earlier package outputs.
    score_tier_summary(y_use, score_use, w_use).to_csv(ROOT / "outputs" / "tables" / f"score_tier_performance_{args.model}_{split_name}.csv", index=False)

    out_tables = ROOT / "outputs" / "tables"
    out_figures = ROOT / "outputs" / "figures"
    out_docs = ROOT / "docs"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_docs.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_tables / "segment_profile_summary.csv", index=False)
    segment_top_categories(profile).to_csv(out_tables / "segment_top_categories.csv", index=False)
    maybe_make_segment_figures(summary, out_figures)

    cluster_scores = None
    cluster_summary = None
    if args.include_clusters:
        cluster_scores, cluster_summary = optional_cluster_segmentation(profile, X_use, args.k_values, args.random_state)
        cluster_scores.to_csv(out_tables / "optional_cluster_silhouette_scores.csv", index=False)
        cluster_summary.to_csv(out_tables / "optional_cluster_profile_summary.csv", index=False)

    write_segmentation_appendix(summary, cluster_summary, out_docs / "segmentation_appendix.md", split_name)
    print(summary.to_string(index=False))
    print(f"Saved segmentation profiles to {out_tables / 'segment_profile_summary.csv'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate score-tier profile segmentation and optional clustering appendix.")
    p.add_argument("--model", default="practical", choices=["practical", "monotonic", "logistic_classweighted", "logistic_population"])
    p.add_argument("--split", default="test", choices=["test", "calibration", "train", "full"], help="Split used to estimate segment positive rates.")
    p.add_argument("--include-clusters", action="store_true", help="Generate optional unsupervised profile clustering appendix.")
    p.add_argument("--k-values", type=int, nargs="+", default=[4, 5], help="Candidate cluster counts for optional clustering.")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--data-zip", default=str(ROOT / "data" / "TakeHomeProject.zip"))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--artifact-dir", default=str(ROOT / "artifacts"))
    p.add_argument("--split-indices", default=str(ROOT / "artifacts" / "shared" / "split_indices.csv"))
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
