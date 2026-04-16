"""
test_features.py
----------------
Unit tests for the feature engineering module.
Run with:  pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_engineering import create_features, encode_categoricals


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Minimal dataframe that mirrors the Telco schema."""
    return pd.DataFrame({
        "tenure":            [1, 24, 60],
        "MonthlyCharges":    [29.85, 56.95, 89.10],
        "TotalCharges":      [29.85, 1397.47, 5374.55],
        "Contract":          ["Month-to-month", "One year", "Two year"],
        "PhoneService":      ["Yes", "No", "Yes"],
        "MultipleLines":     ["No phone service", "No", "Yes"],
        "InternetService":   ["DSL", "DSL", "Fiber optic"],
        "OnlineSecurity":    ["No", "Yes", "No"],
        "OnlineBackup":      ["Yes", "No", "No"],
        "DeviceProtection":  ["No", "Yes", "No"],
        "TechSupport":       ["No", "No", "Yes"],
        "StreamingTV":       ["No", "No", "Yes"],
        "StreamingMovies":   ["No", "No", "No"],
        "Churn":             [0, 0, 1],
    })


# ── Tests ──────────────────────────────────────────────────────────────────────
def test_tenure_group_created(sample_df):
    result = create_features(sample_df)
    assert "tenure_group" in result.columns


def test_tenure_group_values(sample_df):
    result = create_features(sample_df)
    assert set(result["tenure_group"].astype(str)).issubset(
        {"new", "mid", "long-term"}
    )


def test_charges_ratio_created(sample_df):
    result = create_features(sample_df)
    assert "charges_ratio" in result.columns


def test_charges_ratio_is_positive(sample_df):
    result = create_features(sample_df)
    assert (result["charges_ratio"] >= 0).all()


def test_num_services_range(sample_df):
    result = create_features(sample_df)
    assert "num_services" in result.columns
    assert result["num_services"].between(0, 9).all()


def test_is_month_to_month_flag(sample_df):
    result = create_features(sample_df)
    assert result.loc[0, "is_month_to_month"] == 1
    assert result.loc[1, "is_month_to_month"] == 0
    assert result.loc[2, "is_month_to_month"] == 0


def test_no_rows_dropped(sample_df):
    result = create_features(sample_df)
    assert len(result) == len(sample_df)


def test_encode_categoricals_no_objects(sample_df):
    df = create_features(sample_df)
    df = encode_categoricals(df)
    object_cols = df.drop(columns=["Churn"]).select_dtypes(
        include=["object", "category"]
    ).columns
    assert len(object_cols) == 0, f"Still has object cols: {list(object_cols)}"
