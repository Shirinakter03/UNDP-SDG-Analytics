"""
data_loader.py
--------------
Data ingestion, validation, and cleaning pipeline.
Handles missing values, outlier flagging, and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

CLUSTER_LABELS = {
    0: "Vulnerable",
    1: "Developing",
    2: "Advanced",
}

CLUSTER_COLORS = {
    "Vulnerable": "#F87171",
    "Developing": "#FBBF24",
    "Advanced":   "#34D399",
}

FEATURES = ["hdi", "poverty_rate", "internet_access", "co2_per_capita"]

SDG_MAP = {
    "poverty_rate":     "SDG 1 — No Poverty",
    "literacy_rate":    "SDG 4 — Quality Education",
    "internet_access":  "SDG 9 — Industry & Infrastructure",
    "hdi":              "SDG 10 — Reduced Inequalities",
    "co2_per_capita":   "SDG 13 — Climate Action",
}


def load_main(filepath: str = None) -> pd.DataFrame:
    """
    Load and validate the main SDG indicators dataset.

    Cleaning steps applied:
    1. Strip whitespace from string columns
    2. Coerce numeric columns — non-parseable values become NaN
    3. Drop rows with >2 missing numeric indicators (too incomplete)
    4. Fill remaining NaNs with regional median (avoids global bias)
    5. Flag imputed rows for transparency in analysis
    """
    path = filepath or DATA_DIR / "sdg_indicators.csv"
    df = pd.read_csv(path)

    # Step 1 — strip whitespace
    df["country"] = df["country"].str.strip()
    df["region"]  = df["region"].str.strip()

    numeric_cols = ["hdi", "poverty_rate", "literacy_rate",
                    "co2_per_capita", "internet_access", "gni_per_capita"]

    # Step 2 — coerce to numeric
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 3 — drop rows missing more than 2 indicators
    missing_counts = df[numeric_cols].isnull().sum(axis=1)
    dropped = df[missing_counts > 2]["country"].tolist()
    if dropped:
        print(f"[data_loader] Dropped {len(dropped)} row(s) "
              f"due to excessive missingness: {dropped}")
    df = df[missing_counts <= 2].copy()

    # Step 4 — impute remaining NaNs with regional median
    df["_imputed"] = df[numeric_cols].isnull().any(axis=1)
    for col in numeric_cols:
        regional_median = df.groupby("region")[col].transform("median")
        df[col] = df[col].fillna(regional_median)
        # Fallback to global median if regional also NaN
        df[col] = df[col].fillna(df[col].median())

    # Step 5 — sanity bounds check
    assert df["hdi"].between(0, 1).all(), "HDI values out of [0,1] range"
    assert df["poverty_rate"].ge(0).all(), "Negative poverty rate found"
    assert df["internet_access"].between(0, 100).all(), \
        "Internet access % out of [0,100]"

    df = df.reset_index(drop=True)
    print(f"[data_loader] Loaded {len(df)} countries, "
          f"{df['_imputed'].sum()} row(s) had values imputed.")
    return df


def load_trends(filepath: str = None) -> pd.DataFrame:
    """Load HDI trend data (2015–2023)."""
    path = filepath or DATA_DIR / "hdi_trends.csv"
    df = pd.read_csv(path)
    df["hdi"] = pd.to_numeric(df["hdi"], errors="coerce")
    df = df.dropna(subset=["hdi"])
    return df


def attach_cluster_labels(df: pd.DataFrame,
                           labels: pd.Series) -> pd.DataFrame:
    """
    Merge integer cluster IDs with human-readable labels and colours.
    Sorts clusters by mean HDI so label 0 = lowest HDI (Vulnerable).
    """
    df = df.copy()
    df["cluster_id"] = labels.values

    # Sort cluster IDs by ascending mean HDI so labels are consistent
    mean_hdi = df.groupby("cluster_id")["hdi"].mean().sort_values()
    rank_map = {old: new for new, old in enumerate(mean_hdi.index)}
    df["cluster_id"]   = df["cluster_id"].map(rank_map)
    df["cluster"]      = df["cluster_id"].map(CLUSTER_LABELS)
    df["cluster_color"]= df["cluster"].map(CLUSTER_COLORS)
    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cluster summary statistics used in the overview cards.
    Returns mean of key indicators by cluster.
    """
    cols = ["hdi", "poverty_rate", "internet_access",
            "co2_per_capita", "gni_per_capita", "literacy_rate"]
    return (df.groupby("cluster")[cols]
              .mean()
              .round(3)
              .reset_index())
