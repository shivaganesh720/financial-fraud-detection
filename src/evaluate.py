"""
evaluate.py — Model evaluation plots and reports.

All figures are saved to ``outputs/plots/`` using the dark-background style
with cyan/teal accent colours.  ``plt.show()`` is never called.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    f1_score,
)

from src.config import (
    ACCENT_COLOR,
    ACCENT_COLOR_2,
    ACCENT_COLOR_3,
    PLOTS_DIR,
    PLOT_DPI,
    PLOT_STYLE,
)


def _ensure_dirs() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. ROC curves (all models)
# --------------------------------------------------------------------------- #
def plot_roc_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> str:
    """Plot ROC curves for every model on the same axes.

    Parameters
    ----------
    models : Dict[str, estimator]
    X_test, y_test : test data.

    Returns
    -------
    str   Path to saved figure.
    """
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 7))

    palette = [ACCENT_COLOR, ACCENT_COLOR_2, ACCENT_COLOR_3, "#FFD700", "#DA70D6"]
    for idx, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(X_test.values)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=palette[idx % len(palette)],
                label=f"{name}  (AUC = {roc_auc:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, color="white")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved ROC curves → {path}")
    return path


# --------------------------------------------------------------------------- #
# 2. Precision–Recall curves (all models)
# --------------------------------------------------------------------------- #
def plot_pr_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> str:
    """Plot Precision–Recall curves for every model.

    Returns
    -------
    str   Path to saved figure.
    """
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 7))

    palette = [ACCENT_COLOR, ACCENT_COLOR_2, ACCENT_COLOR_3, "#FFD700", "#DA70D6"]
    for idx, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(X_test.values)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = auc(rec, prec)
        ax.plot(rec, prec, color=palette[idx % len(palette)],
                label=f"{name}  (AP = {ap:.4f})", linewidth=2)

    baseline = y_test.mean()
    ax.axhline(y=baseline, linestyle="--", color="gray", linewidth=1, label=f"Baseline ({baseline:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curve Comparison", fontsize=14, color="white")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    path = os.path.join(PLOTS_DIR, "pr_curves.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved PR curves → {path}")
    return path


# --------------------------------------------------------------------------- #
# 3. Confusion matrix (best model)
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Best Model",
) -> str:
    """Confusion matrix heatmap with TP/FP/TN/FN annotations.

    Returns
    -------
    str   Path to saved figure.
    """
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 6))

    y_pred = model.predict(X_test.values)
    cm = confusion_matrix(y_test, y_pred)

    # Annotate with count + label
    tn, fp, fn, tp = cm.ravel()
    annot = np.array([
        [f"TN\n{tn:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn:,}", f"TP\n{tp:,}"],
    ])

    sns.heatmap(
        cm, annot=annot, fmt="", cmap="BuGn",
        xticklabels=["Genuine", "Fraud"],
        yticklabels=["Genuine", "Fraud"],
        ax=ax, linewidths=1, linecolor="black",
        annot_kws={"size": 14, "fontweight": "bold"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, color="white")

    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved confusion matrix → {path}")
    return path


# --------------------------------------------------------------------------- #
# 4. Threshold analysis
# --------------------------------------------------------------------------- #
def plot_threshold_analysis(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Best Model",
) -> str:
    """Plot Precision, Recall, and F1 vs. classification threshold.

    Returns
    -------
    str   Path to saved figure.
    """
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(9, 6))

    y_prob = model.predict_proba(X_test.values)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 200)

    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        p = precision_recall_f1(y_test.values, y_pred_t)
        precisions.append(p[0])
        recalls.append(p[1])
        f1s.append(p[2])

    ax.plot(thresholds, precisions, label="Precision", color=ACCENT_COLOR, linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", color=ACCENT_COLOR_3, linewidth=2)
    ax.plot(thresholds, f1s, label="F1 Score", color=ACCENT_COLOR_2, linewidth=2)

    best_f1_idx = np.argmax(f1s)
    ax.axvline(thresholds[best_f1_idx], linestyle="--", color="white", alpha=0.6,
               label=f"Best F1 threshold = {thresholds[best_f1_idx]:.2f}")

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Threshold Analysis — {model_name}", fontsize=14, color="white")
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    path = os.path.join(PLOTS_DIR, "threshold_analysis.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved threshold analysis → {path}")
    return path


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    """Return (precision, recall, f1) handling zero-division."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    return (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    )


# --------------------------------------------------------------------------- #
# 5. Classification report
# --------------------------------------------------------------------------- #
def print_classification_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Best Model",
) -> str:
    """Print the sklearn classification report to stdout.

    Returns
    -------
    str   The report text.
    """
    y_pred = model.predict(X_test.values)
    report = classification_report(
        y_test, y_pred, target_names=["Genuine", "Fraud"], digits=4,
    )
    print(f"\nClassification Report — {model_name}")
    print(report)
    return report


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_evaluation(
    models: Dict[str, Any],
    best_name: str,
    best_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Execute all evaluation plots and the classification report.

    Parameters
    ----------
    models : Dict[str, estimator]
        All fitted models.
    best_name : str
        Name of the best model.
    best_model : estimator
        The best fitted model.
    X_test, y_test : test data.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    plot_roc_curves(models, X_test, y_test)
    plot_pr_curves(models, X_test, y_test)
    plot_confusion_matrix(best_model, X_test, y_test, model_name=best_name)
    plot_threshold_analysis(best_model, X_test, y_test, model_name=best_name)
    print_classification_report(best_model, X_test, y_test, model_name=best_name)

    print("[evaluate] Evaluation complete.\n")
