"""
feature_engineering.py
-----------------------
All feature engineering for the churn pipeline:

  1. Create new features from existing columns
  2. Encode categorical variables
  3. Scale numeric features
  4. Split into train/test
  5. Apply SMOTE to handle class imbalance on training set
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


# ── Constants ──────────────────────────────────────────────────────────────────
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges",
                "charges_ratio", "num_services"]

RANDOM_STATE = 42


# ── Step 1: Feature Creation ───────────────────────────────────────────────────
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new predictive features from raw columns.

    New columns added:
      - tenure_group       : categorical bucket (new / mid / long-term)
      - charges_ratio      : MonthlyCharges / (TotalCharges + 1)
      - num_services       : count of active services
      - is_month_to_month  : 1 if contract is month-to-month

    Parameters
    ----------
    df : pd.DataFrame  (cleaned, from data_loader)

    Returns
    -------
    pd.DataFrame with new feature columns appended
    """
    df = df.copy()

    # Tenure buckets — captures non-linear churn pattern
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 36, 72],
        labels=["new", "mid", "long-term"],
        include_lowest=True,
    )

    # Charges ratio — new customers have high monthly vs low total
    df["charges_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # Service engagement score
    service_flag = df[SERVICE_COLS].apply(
        lambda col: col.map(lambda v: 1 if v not in ["No", "No phone service",
                                                      "No internet service"] else 0)
    )
    df["num_services"] = service_flag.sum(axis=1)

    # High-churn contract type flag
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    print(f"[feature_engineering] Created 4 new features")
    return df


# ── Step 2: Encoding ───────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode all object/category columns (except target).
    Binary columns (Yes/No) become 0/1.
    Multi-class columns get integer codes.
    """
    df = df.copy()
    le = LabelEncoder()

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col == "Churn":
            continue
        df[col] = le.fit_transform(df[col].astype(str))

    print(f"[feature_engineering] Encoded categorical columns")
    return df


# ── Step 3: Full preprocessing pipeline ───────────────────────────────────────
def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    apply_smote: bool = True,
) -> tuple:
    """
    Full preprocessing: feature creation → encoding → split → scale → SMOTE.

    Parameters
    ----------
    df         : cleaned dataframe from data_loader
    test_size  : fraction held out for test set
    apply_smote: whether to apply SMOTE on training data

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # 1. Feature creation + encoding
    df = create_features(df)
    df = encode_categoricals(df)

    # 2. Split features / target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_names = X.columns.tolist()

    # 3. Train / test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    # 4. Fill any remaining NaNs before scaling/SMOTE
    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)

    # 4. Scale numeric columns only
    scaler = StandardScaler()
    #X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    #X_test[NUMERIC_COLS]  = scaler.transform(X_test[NUMERIC_COLS])

    print(f"[feature_engineering] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[feature_engineering] Train churn rate before SMOTE: "
          f"{y_train.mean():.1%}")

    # 5. SMOTE — oversample minority class on TRAINING set only
    if apply_smote:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"[feature_engineering] After SMOTE: {X_train.shape} "
              f"| churn rate: {y_train.mean():.1%}")

    return X_train, X_test, y_train, y_test, scaler, feature_names
