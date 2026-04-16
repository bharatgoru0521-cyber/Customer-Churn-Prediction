"""
evaluate.py
-----------
Evaluate all fitted models and produce:

  - Classification report (precision, recall, F1)
  - Confusion matrix heatmap
  - ROC curve (all models on one plot)
  - Feature importance (Random Forest + Logistic Regression coefficients)
  - Summary metrics table

All plots saved to outputs/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

OUTPUT_DIR = "outputs"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── 1. Classification Report ───────────────────────────────────────────────────
def print_report(model, X_test, y_test, name: str) -> dict:
    """Print classification report and return key metrics as dict."""
    y_pred = model.predict(X_test)

    print(f"\n── {name} ─────────────────────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall"   : round(recall_score(y_test, y_pred), 4),
        "F1"       : round(f1_score(y_test, y_pred), 4),
        "ROC-AUC"  : round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4),
    }


# ── 2. Confusion Matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test, name: str) -> None:
    """Plot and save a labelled confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13, pad=12)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe = name.lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ── 3. ROC Curve (all models) ─────────────────────────────────────────────────
def plot_roc_curves(fitted_models: dict, X_test, y_test) -> None:
    """
    Plot ROC curves for all models on a single axes.
    Each model needs predict_proba support.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for (name, model), color in zip(fitted_models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=10)

    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ── 4. Feature Importance ──────────────────────────────────────────────────────
def plot_feature_importance(
    fitted_models: dict,
    feature_names: list,
    top_n: int = 15,
) -> None:
    """
    Plot feature importance for Random Forest and
    coefficient magnitude for Logistic Regression.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, model in fitted_models.items():
        if name == "Random Forest":
            importances = model.feature_importances_
            title = "Random Forest — Feature Importances"
            xlabel = "Importance (mean decrease in impurity)"
        elif name == "Logistic Regression":
            importances = np.abs(model.coef_[0])
            title = "Logistic Regression — |Coefficient| Magnitude"
            xlabel = "|Coefficient|"
        else:
            continue  # Decision Tree skipped (covered by RF)

        indices = np.argsort(importances)[-top_n:]
        features = [feature_names[i] for i in indices]
        values = importances[indices]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(features, values, color="#4C72B0", edgecolor="white")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=13, pad=12)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

        safe = name.lower().replace(" ", "_")
        path = os.path.join(OUTPUT_DIR, f"feature_importance_{safe}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved → {path}")


# ── 5. Threshold Tuning ────────────────────────────────────────────────────────
def plot_threshold_tuning(model, X_test, y_test, name: str = "Random Forest") -> None:
    """
    Plot Precision and Recall vs decision threshold.
    Helps choose threshold based on business cost of errors.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 80)

    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#4C72B0", lw=2)
    ax.plot(thresholds, recalls,   label="Recall",    color="#DD8452", lw=2)
    ax.plot(thresholds, f1s,       label="F1",        color="#55A868", lw=2, ls="--")
    ax.axvline(0.5, color="gray", ls=":", lw=1, label="Default threshold (0.5)")

    best_t = thresholds[np.argmax(f1s)]
    ax.axvline(best_t, color="#C44E52", ls="--", lw=1.5,
               label=f"Best F1 threshold ({best_t:.2f})")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall / F1 vs Threshold — {name}", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)

    path = os.path.join(OUTPUT_DIR, "threshold_tuning.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")
    print(f"  Best F1 threshold: {best_t:.2f}")


# ── 6. Summary Table ───────────────────────────────────────────────────────────
def summary_table(results: list) -> pd.DataFrame:
    """
    Print and return a formatted summary DataFrame of all model metrics.

    Parameters
    ----------
    results : list of dicts from print_report()
    """
    df = pd.DataFrame(results).set_index("Model")
    print("\n── Model Comparison Summary ─────────────────────────────────────")
    print(df.to_string())
    path = os.path.join(OUTPUT_DIR, "model_summary.csv")
    df.to_csv(path)
    print(f"\n  Saved → {path}")
    return df


# ── 7. Run all evaluations ─────────────────────────────────────────────────────
def evaluate_all(
    fitted_models: dict,
    X_test,
    y_test,
    feature_names: list,
) -> pd.DataFrame:
    """
    Run the full evaluation suite for every model.

    Returns
    -------
    pd.DataFrame  summary of all metrics
    """
    results = []
    print("\n══ EVALUATION ══════════════════════════════════════════════════\n")

    for name, model in fitted_models.items():
        metrics = print_report(model, X_test, y_test, name)
        results.append(metrics)
        plot_confusion_matrix(model, X_test, y_test, name)

    print("\n── ROC curves ──────────────────────────────────────────────────")
    plot_roc_curves(fitted_models, X_test, y_test)

    print("\n── Feature importance ──────────────────────────────────────────")
    plot_feature_importance(fitted_models, feature_names)

    print("\n── Threshold tuning (Random Forest) ────────────────────────────")
    plot_threshold_tuning(
        fitted_models["Random Forest"], X_test, y_test, "Random Forest"
    )

    return summary_table(results)
