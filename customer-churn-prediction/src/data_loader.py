"""
data_loader.py
--------------
Loads and cleans the Telco Customer Churn dataset.

Steps:
  1. Read CSV
  2. Fix TotalCharges (stored as string in raw data)
  3. Drop customerID (not predictive)
  4. Encode binary target (Yes/No -> 1/0)
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw Telco churn CSV and apply basic cleaning.

    Parameters
    ----------
    filepath : str
        Path to WA_Fn-UseC_-Telco-Customer-Churn.csv

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    df = pd.read_csv(filepath)

    # ── Fix TotalCharges: raw data has spaces instead of 0 for new customers ──
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # ── Drop non-predictive ID column ──
    df.drop(columns=["customerID"], inplace=True)

    # ── Encode target: Yes -> 1, No -> 0 ──
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print(f"[data_loader] Loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"[data_loader] Churn rate: {df['Churn'].mean():.1%}")

    return df


def summarize(df: pd.DataFrame) -> None:
    """Print a quick data summary."""
    print("\n── Shape ──────────────────────────────")
    print(f"  Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

    print("\n── Missing values ─────────────────────")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "  None")

    print("\n── Target distribution ────────────────")
    counts = df["Churn"].value_counts()
    print(f"  No churn : {counts[0]:,}  ({counts[0]/len(df):.1%})")
    print(f"  Churned  : {counts[1]:,}  ({counts[1]/len(df):.1%})")

    print("\n── Dtypes ─────────────────────────────")
    print(df.dtypes.value_counts().to_string())
