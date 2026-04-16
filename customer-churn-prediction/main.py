"""
main.py
-------
Run the complete Customer Churn Prediction pipeline:

    python main.py

Steps:
  1. Load & clean data
  2. Feature engineering + SMOTE
  3. Train all 3 models
  4. Evaluate & save plots
"""

import sys
import os

# ── Allow imports from src/ ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, summarize
from src.feature_engineering import preprocess
from src.train import train_all
from src.evaluate import evaluate_all

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():
    print("=" * 60)
    print("  Customer Churn Prediction Pipeline")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────
    print("\n[1/4] Loading data...")
    df = load_data(DATA_PATH)
    summarize(df)

    # ── Step 2: Feature engineering ────────────────────────────────
    print("\n[2/4] Feature engineering & preprocessing...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(
        df,
        test_size=0.2,
        apply_smote=True,
    )

    # ── Step 3: Train models ───────────────────────────────────────
    print("\n[3/4] Training models...")
    fitted_models = train_all(X_train, y_train, cv_folds=5)

    # ── Step 4: Evaluate ───────────────────────────────────────────
    print("\n[4/4] Evaluating models...")
    summary = evaluate_all(fitted_models, X_test, y_test, feature_names)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  Check the outputs/ folder for all plots and the model summary.")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
