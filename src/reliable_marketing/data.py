from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd


def resolve_data_dir(data_zip: str | Path | None = None, data_dir: str | Path | None = None) -> Path:
    """Return a directory containing census-bureau.data and census-bureau.columns.

    If data_dir is provided and valid, use it. Otherwise extract data_zip next to itself.
    """
    if data_dir is not None:
        dd = Path(data_dir)
        if (dd / "census-bureau.data").exists() and (dd / "census-bureau.columns").exists():
            return dd
    if data_zip is None:
        raise FileNotFoundError("Provide --data-dir or --data-zip")
    zpath = Path(data_zip)
    if not zpath.exists():
        raise FileNotFoundError(f"Missing data zip: {zpath}")
    extract_dir = zpath.parent / "TakeHomeProject"
    if not (extract_dir / "census-bureau.data").exists():
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(zpath.parent)
    # Some zips contain a top-level TakeHomeProject folder, some may not.
    if (extract_dir / "census-bureau.data").exists():
        return extract_dir
    if (zpath.parent / "census-bureau.data").exists():
        return zpath.parent
    raise FileNotFoundError("Could not locate census-bureau.data after extraction")


def load_census_income(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    data_dir = Path(data_dir)
    columns = [line.strip() for line in open(data_dir / "census-bureau.columns", encoding="utf-8")]
    df = pd.read_csv(data_dir / "census-bureau.data", header=None, names=columns, skipinitialspace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    y = df["label"].eq("50000+.").astype(int)
    weights = df["weight"].astype(float)
    X = df.drop(columns=["label", "weight"])
    return X, y, weights
