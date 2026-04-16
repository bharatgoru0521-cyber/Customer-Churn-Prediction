"""
train.py
--------
Train and persist all three models:
  - Logistic Regression
  - Decision Tree
  - Random Forest

Each model is returned as a fitted sklearn estimator.
Models are saved to outputs/ via joblib.
"""

import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"


def get_models() -> dict:
    """
    Return a dictionary of named, configured (but unfitted) models.

    Hyperparameters are sensible defaults — good starting point
    before any tuning.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,                   # inverse regularization strength
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,             # prevent overfitting
            min_samples_leaf=20,     # require meaningful leaf size
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,        # enough trees to stabilize variance
            max_depth=10,
            min_samples_leaf=10,
            n_jobs=-1,               # use all CPU cores
            random_state=RANDOM_STATE,
        ),
    }


def train_all(X_train, y_train, cv_folds: int = 5) -> dict:
    """
    Fit all models on training data with cross-validation scoring.

    Parameters
    ----------
    X_train   : feature matrix (scaled, SMOTE-balanced)
    y_train   : target vector
    cv_folds  : number of CV folds for validation

    Returns
    -------
    dict  { model_name: fitted_estimator }
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = get_models()
    fitted = {}

    print("\n── Training models ─────────────────────────────────────────────")

    for name, model in models.items():
        print(f"\n  [{name}]")

        # Cross-validation on training data (F1 — better for imbalanced)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring="f1", n_jobs=-1
        )
        print(f"    CV F1 scores : {cv_scores.round(3)}")
        print(f"    Mean CV F1   : {cv_scores.mean():.3f} "
              f"(± {cv_scores.std():.3f})")

        # Final fit on full training set
        model.fit(X_train, y_train)
        fitted[name] = model

        # Save model to disk
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(OUTPUT_DIR, f"{safe_name}.joblib")
        joblib.dump(model, path)
        print(f"    Saved → {path}")

    print("\n── All models trained and saved ────────────────────────────────\n")
    return fitted


def load_model(name: str):
    """Load a saved model by name (e.g. 'random_forest')."""
    path = os.path.join(OUTPUT_DIR, f"{name}.joblib")
    return joblib.load(path)
