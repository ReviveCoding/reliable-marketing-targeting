from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

FINANCIAL_COLS = ["wage per hour", "capital gains", "capital losses", "dividends from stocks"]
ADV_DEGREES = {
    "Bachelors degree",
    "Masters degree",
    "Prof school degree",
    "Doctorate degree",
    "Bachelors degree(BA AB BS)",
    "Masters degree(MA MS MEng MEd MSW MBA)",
    "Prof school degree (MD DDS DVM LLB JD)",
    "Doctorate degree(PhD EdD)",
}
PROFESSIONAL_OCC_SUBSTRINGS = [
    "Professional specialty",
    "Executive admin and managerial",
    "Technicians and related support",
    "Sales",
]


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip().replace({"?": "Unknown"})
    return out


def build_pht_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the selected PHT-inspired/economic-lifecycle feature layer.

    The original row-level population weight is intentionally not part of this input.
    """
    d = clean_categories(df)
    d["age_squared"] = d["age"] ** 2
    d["working_age_flag"] = d["age"].between(18, 64).astype(int)
    d["age_bin"] = pd.cut(
        d["age"],
        [-1, 17, 24, 34, 44, 54, 64, 200],
        labels=["0_17", "18_24", "25_34", "35_44", "45_54", "55_64", "65_plus"],
    ).astype(str)
    d["full_year_worker"] = (d["weeks worked in year"] >= 48).astype(int)
    d["partial_year_worker"] = ((d["weeks worked in year"] > 0) & (d["weeks worked in year"] < 48)).astype(int)
    d["not_worked_flag"] = (d["weeks worked in year"] == 0).astype(int)
    d["employment_intensity"] = d["weeks worked in year"] / 52.0

    for col in FINANCIAL_COLS:
        d[f"has_{col}"] = (d[col] > 0).astype(int)
        d[f"log1p_{col}"] = np.log1p(d[col].clip(lower=0))

    d["any_investment_signal"] = (
        (d["capital gains"] > 0) | (d["capital losses"] > 0) | (d["dividends from stocks"] > 0)
    ).astype(int)
    d["employer_size_bin"] = pd.cut(
        d["num persons worked for employer"],
        [-1, 0, 1, 3, 6, 99],
        labels=["0", "1", "2_3", "4_6", "7_plus"],
    ).astype(str)
    d["self_employed_flag"] = (d["own business or self employed"] > 0).astype(int)

    for col in df.select_dtypes(include="object").columns:
        d[col + "_is_unknown"] = df[col].astype(str).str.strip().isin(["?", "Unknown"]).astype(int)

    d["birth_self_group"] = np.where(
        d["country of birth self"].eq("United-States"),
        "US",
        np.where(d["country of birth self"].isin(["Unknown", "?"]), "Unknown", "Foreign"),
    )
    d["marital_tax_combo"] = d["marital stat"].astype(str) + "__" + d["tax filer stat"].astype(str)
    d["agebin_education_combo"] = d["age_bin"].astype(str) + "__" + d["education"].astype(str)
    return d


def build_pht_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Optional compact v2 features used in the A/B/C ablation package."""
    d = build_pht_features(df)
    occ = d["major occupation code"].astype(str)
    d["college_plus"] = d["education"].isin(ADV_DEGREES).astype(int)
    d["prime_age_flag"] = d["age"].between(30, 55).astype(int)
    d["professional_occ_flag"] = occ.apply(lambda x: any(s in x for s in PROFESSIONAL_OCC_SUBSTRINGS)).astype(int)
    d["stable_professional_flag"] = (
        (d["college_plus"] == 1) & (d["full_year_worker"] == 1) & (d["professional_occ_flag"] == 1)
    ).astype(int)
    d["business_owner_investor_flag"] = (
        (d["self_employed_flag"] == 1) & (d["any_investment_signal"] == 1)
    ).astype(int)
    d["retired_asset_signal"] = (
        (d["age"] >= 60) & (d["weeks worked in year"] < 20) & (d["any_investment_signal"] == 1)
    ).astype(int)
    d["low_income_structural_flag"] = (
        (d["not_worked_flag"] == 1) & (~d["education"].isin(ADV_DEGREES)) & (d["any_investment_signal"] == 0)
    ).astype(int)
    d["income_readiness_score_v2"] = (
        1.0 * d["college_plus"]
        + 1.0 * d["full_year_worker"]
        + 0.75 * d["prime_age_flag"]
        + 0.75 * d["professional_occ_flag"]
        + 0.75 * d["any_investment_signal"]
        + 0.35 * d["self_employed_flag"]
        - 0.50 * d["low_income_structural_flag"]
    )
    d["career_stage"] = d["age_bin"].astype(str) + "__" + np.where(
        d["full_year_worker"].eq(1), "full_year", np.where(d["partial_year_worker"].eq(1), "partial", "not_worked")
    )
    return d


def fit_rare_category_maps(X_train_fe: pd.DataFrame, min_count: int = 150) -> Dict[str, Set[str]]:
    maps: Dict[str, Set[str]] = {}
    for col in X_train_fe.select_dtypes(include="object").columns:
        vc = X_train_fe[col].astype(str).value_counts()
        maps[col] = set(vc[vc >= min_count].index.astype(str))
    return maps


def apply_rare_category_maps(X_fe: pd.DataFrame, maps: Dict[str, Set[str]]) -> pd.DataFrame:
    out = X_fe.copy()
    for col, keep in maps.items():
        out[col] = out[col].astype(str).where(out[col].astype(str).isin(keep), "Other_Rare")
    return out


@dataclass
class FeatureEncoder:
    rare_maps: Dict[str, Set[str]]
    categorical_cols: list[str]
    numeric_cols: list[str]
    ordinal_encoder: OrdinalEncoder
    feature_names: list[str]
    feature_version: str = "pht"

    @classmethod
    def fit(cls, X_train: pd.DataFrame, min_count: int = 150, feature_version: str = "pht") -> "FeatureEncoder":
        fe = build_pht_v2_features(X_train) if feature_version == "pht_v2" else build_pht_features(X_train)
        maps = fit_rare_category_maps(fe, min_count=min_count)
        fe = apply_rare_category_maps(fe, maps)
        cat_cols = fe.select_dtypes(include="object").columns.tolist()
        num_cols = fe.select_dtypes(exclude="object").columns.tolist()
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if cat_cols:
            enc.fit(fe[cat_cols].astype(str))
        return cls(
            rare_maps=maps,
            categorical_cols=cat_cols,
            numeric_cols=num_cols,
            ordinal_encoder=enc,
            feature_names=num_cols + cat_cols,
            feature_version=feature_version,
        )

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        fe = build_pht_v2_features(X) if self.feature_version == "pht_v2" else build_pht_features(X)
        fe = apply_rare_category_maps(fe, self.rare_maps)
        X_num = fe[self.numeric_cols].to_numpy(dtype=float)
        if self.categorical_cols:
            X_cat = self.ordinal_encoder.transform(fe[self.categorical_cols].astype(str))
        else:
            X_cat = np.empty((len(fe), 0))
        return np.hstack([X_num, X_cat]).astype(np.float32)
