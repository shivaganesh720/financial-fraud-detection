"""
explainability.py — SHAP-based model explanations.

Handles both tree-based models (TreeExplainer) and linear models
(LinearExplainer).  All plots are saved, never shown.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from src.config import (
    ACCENT_COLOR,
    OUTPUTS_DIR,
    PLOTS_DIR,
    PLOT_DPI,
    PLOT_STYLE,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def _ensure_dirs() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _get_explainer(
    model: Any,
    X_background: pd.DataFrame,
) -> shap.Explainer:
    """Return the appropriate SHAP explainer for the model type.

    Parameters
    ----------
    model : estimator
        Fitted scikit-learn / XGBoost model.
    X_background : pd.DataFrame
        Background dataset used by KernelExplainer or LinearExplainer.

    Returns
    -------
    shap.Explainer
    """
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, X_background)
    else:
        # Works for RandomForest and XGBoost
        return shap.TreeExplainer(model)


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> tuple[np.ndarray, shap.Explainer]:
    """Compute SHAP values for a (sampled) feature matrix.

    Parameters
    ----------
    model : fitted estimator
    X : pd.DataFrame
        Feature matrix to explain.
    max_samples : int
        Cap on the number of samples (for speed).

    Returns
    -------
    tuple[np.ndarray, shap.Explainer]
        SHAP values array (n_samples × n_features) and the explainer.
    """
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X.copy()

    explainer = _get_explainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers TreeExplainer may return a list of two arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive-class SHAP values

    return shap_values, explainer, X_sample


# --------------------------------------------------------------------------- #
# 1. SHAP Summary plot
# --------------------------------------------------------------------------- #
def plot_shap_summary(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> str:
    """Generate and save a SHAP beeswarm summary plot.

    Parameters
    ----------
    model : fitted estimator
    X : pd.DataFrame
        Feature matrix (ideally the test set).

    Returns
    -------
    str   Path to saved figure.
    """
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)

    print("[explainability] Computing SHAP values for summary plot …")
    shap_values, _, X_sample = compute_shap_values(model, X, max_samples)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title("SHAP Summary — Feature Impact on Fraud Prediction", fontsize=13, color="white")

    path = os.path.join(PLOTS_DIR, "shap_summary.png")
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close("all")
    print(f"[explainability] Saved SHAP summary plot → {path}")
    return path


# --------------------------------------------------------------------------- #
# 2. SHAP force plot (single prediction → HTML)
# --------------------------------------------------------------------------- #
def plot_shap_force(
    model: Any,
    X: pd.DataFrame,
    index: int = 0,
) -> str:
    """Generate a SHAP force plot for a single observation and save as HTML.

    Parameters
    ----------
    model : fitted estimator
    X : pd.DataFrame
    index : int
        Row index inside X to explain.

    Returns
    -------
    str   Path to saved HTML file.
    """
    _ensure_dirs()
    print(f"[explainability] Generating SHAP force plot for sample index {index} …")

    explainer = _get_explainer(model, X.iloc[:200])
    single = X.iloc[[index]]
    sv = explainer.shap_values(single)

    if isinstance(sv, list):
        sv = sv[1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    force_html = shap.force_plot(
        expected_value,
        sv[0],
        single.iloc[0],
        matplotlib=False,
    )

    path = os.path.join(OUTPUTS_DIR, "shap_force_plot.html")
    shap.save_html(path, force_html)
    print(f"[explainability] Saved SHAP force plot → {path}")
    return path


# --------------------------------------------------------------------------- #
# 3. Feature importance bar chart from SHAP
# --------------------------------------------------------------------------- #
def plot_shap_importance(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 500,
    top_n: int = 20,
) -> str:
    """Bar chart of mean |SHAP value| per feature.

    Parameters
    ----------
    model : fitted estimator
    X : pd.DataFrame
    max_samples : int
    top_n : int
        How many features to display.

    Returns
    -------
    str   Path to saved figure.
    """
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)

    print("[explainability] Computing SHAP feature importance bar chart …")
    shap_values, _, X_sample = compute_shap_values(model, X, max_samples)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X_sample.columns).sort_values(ascending=True)
    top = importance.tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top.index, top.values, color=ACCENT_COLOR, edgecolor="white", linewidth=0.4)
    ax.set_title(f"Top {top_n} Features by Mean |SHAP Value|", fontsize=14, color="white")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)

    path = os.path.join(PLOTS_DIR, "shap_feature_importance.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[explainability] Saved SHAP importance bar chart → {path}")
    return path


# --------------------------------------------------------------------------- #
# 4. Explain a single transaction
# --------------------------------------------------------------------------- #
def explain_transaction(
    model: Any,
    feature_values: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> Dict[str, float]:
    """Return SHAP contributions for a single transaction.

    Parameters
    ----------
    model : fitted estimator
    feature_values : pd.DataFrame
        1-row DataFrame with feature values.
    feature_names : list[str] | None
        Optional override for column names.

    Returns
    -------
    Dict[str, float]
        Feature name → SHAP value.
    """
    explainer = _get_explainer(model, feature_values)
    sv = explainer.shap_values(feature_values)
    if isinstance(sv, list):
        sv = sv[1]

    names = feature_names or feature_values.columns.tolist()
    contributions = dict(zip(names, sv[0]))
    return contributions


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_explainability(
    model: Any,
    X_test: pd.DataFrame,
    model_name: str = "Best Model",
) -> None:
    """Run all SHAP explanation routines.

    Parameters
    ----------
    model : fitted estimator
    X_test : pd.DataFrame
    model_name : str
    """
    print("\n" + "=" * 60)
    print(f"EXPLAINABILITY  ({model_name})")
    print("=" * 60)

    plot_shap_summary(model, X_test)
    plot_shap_importance(model, X_test)
    plot_shap_force(model, X_test, index=0)

    # Explain first fraud in test set (if any)
    print("\n[explainability] Example single-transaction explanation:")
    sample = X_test.iloc[[0]]
    contribs = explain_transaction(model, sample)
    top_5 = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    for feat, val in top_5:
        direction = "↑ fraud" if val > 0 else "↓ genuine"
        print(f"  {feat:>12s}  SHAP = {val:+.4f}  ({direction})")

    print("\n[explainability] Explainability complete.\n")
