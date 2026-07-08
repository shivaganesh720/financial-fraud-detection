"""
eda.py — Exploratory Data Analysis for the credit-card fraud dataset.

Every plotting function saves its figure to ``outputs/plots/`` and never
calls ``plt.show()``.  All plots use the dark_background style with
cyan / teal accent colours.

Fixes vs original
-----------------
- Amount distribution: x-axis range now computed from the data rather than
  hard-capped at 2000, so fraud bars (which can reach $3000+) are visible.
- Time distribution: adds a small-dataset guard and note when fraud count < 30,
  because sparse fraud points produce misleading spike patterns.
- New: plot_fraud_amount_stats() — side-by-side mean/median/max amount bars.
"""

from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    ACCENT_COLOR,
    ACCENT_COLOR_2,
    ACCENT_COLOR_3,
    PLOTS_DIR,
    PLOT_DPI,
    PLOT_STYLE,
    RANDOM_STATE,
)


def _ensure_dirs() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame) -> str:
    """Bar chart of fraud vs genuine with counts and percentages."""
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = df["Class"].value_counts().sort_index()
    labels = ["Genuine (0)", "Fraud (1)"]
    colors = [ACCENT_COLOR, ACCENT_COLOR_3]
    bars   = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.6)

    total = len(df)
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({pct:.3f}%)",
            ha="center", va="bottom", fontsize=11, color="white", fontweight="bold",
        )

    ax.set_title("Class Distribution — Fraud vs Genuine", fontsize=14, color="white")
    ax.set_ylabel("Count", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.15)

    path = os.path.join(PLOTS_DIR, "class_distribution.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved class distribution plot → {path}")
    return path


# ---------------------------------------------------------------------------
# 2. Correlation heatmap (top features)
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, top_n: int = 15) -> str:
    """Correlation heatmap of the top_n features most correlated with Class."""
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)

    corr_with_class = (
        df.corr(numeric_only=True)["Class"]
        .drop("Class").abs()
        .sort_values(ascending=False)
    )
    top_features = corr_with_class.head(top_n).index.tolist() + ["Class"]
    corr_matrix  = df[top_features].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(180, 10, as_cmap=True)
    sns.heatmap(
        corr_matrix, mask=mask, cmap=cmap, center=0,
        annot=True, fmt=".2f", linewidths=0.5, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title(f"Correlation Heatmap — Top {top_n} Features", fontsize=14, color="white")

    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved correlation heatmap → {path}")
    return path


# ---------------------------------------------------------------------------
# 3. Amount distribution  (FIXED: dynamic x-axis range)
# ---------------------------------------------------------------------------

def plot_amount_distribution(df: pd.DataFrame) -> str:
    """
    Overlapping histograms of transaction amount for fraud vs genuine.

    FIX: The original hard-coded x-axis cap of $2,000 hid all fraud bars
    when using the mock dataset (fraud amounts: $500–$3,000).  The range
    is now computed from the 99th percentile of the full dataset so that
    both classes are always visible.
    """
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))

    genuine = df[df["Class"] == 0]["Amount"]
    fraud   = df[df["Class"] == 1]["Amount"]

    # Dynamic upper bound: 99th percentile of all transactions (not hard-coded)
    upper = float(np.percentile(df["Amount"], 99))
    upper = max(upper, fraud.max() if len(fraud) > 0 else upper) * 1.05

    ax.hist(genuine, bins=80, alpha=0.6, label="Genuine",
            color=ACCENT_COLOR,  range=(0, upper))
    ax.hist(fraud,   bins=40, alpha=0.8, label="Fraud",
            color=ACCENT_COLOR_3, range=(0, upper))

    ax.set_title("Transaction Amount Distribution — Fraud vs Genuine",
                 fontsize=14, color="white")
    ax.set_xlabel("Amount ($)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=11)

    path = os.path.join(PLOTS_DIR, "amount_distribution.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved amount distribution plot → {path}")
    return path


# ---------------------------------------------------------------------------
# 4. Time distribution  (FIXED: small-dataset guard)
# ---------------------------------------------------------------------------

def plot_time_distribution(df: pd.DataFrame) -> str:
    """
    Overlapping histograms of time for fraud vs genuine.

    FIX: On small / mock datasets the fraud class may have fewer than 30
    rows, causing misleading isolated spikes that look like temporal patterns.
    A subtitle note is now added when fraud_n < 30.
    """
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))

    genuine   = df[df["Class"] == 0]["Time"]
    fraud     = df[df["Class"] == 1]["Time"]
    fraud_n   = len(fraud)

    ax.hist(genuine, bins=60, alpha=0.6, label="Genuine",
            color=ACCENT_COLOR,   density=True)
    ax.hist(fraud,   bins=60, alpha=0.7, label="Fraud",
            color=ACCENT_COLOR_3, density=True)

    ax.set_title("Transaction Time Distribution — Fraud vs Genuine",
                 fontsize=14, color="white")
    ax.set_xlabel("Time (seconds from first transaction)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)

    if fraud_n < 30:
        ax.text(
            0.99, 0.97,
            f"⚠ Only {fraud_n} fraud samples — spikes may not reflect real patterns.",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=ACCENT_COLOR_3, style="italic",
        )

    path = os.path.join(PLOTS_DIR, "time_distribution.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved time distribution plot → {path}")
    return path


# ---------------------------------------------------------------------------
# 5. Box plots for top PCA features
# ---------------------------------------------------------------------------

def plot_pca_boxplots(df: pd.DataFrame, top_n: int = 10) -> str:
    """Box plots of the top top_n PCA features for fraud vs genuine."""
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)

    pca_cols  = [c for c in df.columns if c.startswith("V")]
    corr      = (
        df[pca_cols + ["Class"]]
        .corr(numeric_only=True)["Class"]
        .drop("Class").abs()
        .sort_values(ascending=False)
    )
    top_feats = corr.head(top_n).index.tolist()

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for idx, feat in enumerate(top_feats):
        ax = axes[idx]
        data_genuine = df[df["Class"] == 0][feat]
        data_fraud   = df[df["Class"] == 1][feat]
        bp = ax.boxplot(
            [data_genuine, data_fraud],
            tick_labels=["Genuine", "Fraud"],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=1.5),
            flierprops=dict(marker=".", markerfacecolor=ACCENT_COLOR_2,
                            markersize=2, alpha=0.4),
        )
        bp["boxes"][0].set_facecolor(ACCENT_COLOR);  bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(ACCENT_COLOR_3); bp["boxes"][1].set_alpha(0.7)
        ax.set_title(feat, fontsize=11, color="white")

    fig.suptitle("Box Plots — Top 10 PCA Features (Fraud vs Genuine)",
                 fontsize=15, color="white", y=1.01)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "pca_boxplots.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved PCA box plots → {path}")
    return path


# ---------------------------------------------------------------------------
# 6. Feature importance from a quick Random Forest
# ---------------------------------------------------------------------------

def plot_feature_importance(X: pd.DataFrame, y: pd.Series, top_n: int = 20) -> str:
    """Train a lightweight Random Forest and plot feature importances."""
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)

    print("[eda] Training quick Random Forest for feature importance…")
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X, y)

    importances = (
        pd.Series(rf.feature_importances_, index=X.columns)
        .sort_values(ascending=True)
    )
    top = importances.tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top.index, top.values, color=ACCENT_COLOR, edgecolor="white", linewidth=0.4)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
                 fontsize=14, color="white")
    ax.set_xlabel("Importance", fontsize=12)

    path = os.path.join(PLOTS_DIR, "feature_importance_rf.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved feature importance plot → {path}")
    return path


# ---------------------------------------------------------------------------
# 7. NEW — Fraud vs Genuine amount statistics
# ---------------------------------------------------------------------------

def plot_fraud_amount_stats(df: pd.DataFrame) -> str:
    """
    Side-by-side grouped bar chart comparing mean, median, and max
    transaction amount for fraud vs genuine classes.

    This is the clearest way to see that fraud transactions in this dataset
    tend to cluster at much higher dollar values than genuine ones.
    """
    _ensure_dirs()
    plt.style.use(PLOT_STYLE)

    genuine = df[df["Class"] == 0]["Amount"]
    fraud   = df[df["Class"] == 1]["Amount"]

    stats = {
        "Mean":   [genuine.mean(),   fraud.mean()],
        "Median": [genuine.median(), fraud.median()],
        "Max":    [genuine.max(),    fraud.max()],
    }

    x      = np.arange(len(stats))
    width  = 0.35
    labels = list(stats.keys())

    fig, ax = plt.subplots(figsize=(9, 6))

    genuine_vals = [v[0] for v in stats.values()]
    fraud_vals   = [v[1] for v in stats.values()]

    bars_g = ax.bar(x - width / 2, genuine_vals, width,
                    label="Genuine", color=ACCENT_COLOR,  alpha=0.85, edgecolor="white")
    bars_f = ax.bar(x + width / 2, fraud_vals,   width,
                    label="Fraud",   color=ACCENT_COLOR_3, alpha=0.85, edgecolor="white")

    for bar in bars_g:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"${bar.get_height():,.0f}", ha="center", va="bottom",
                fontsize=9, color="white")
    for bar in bars_f:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"${bar.get_height():,.0f}", ha="center", va="bottom",
                fontsize=9, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title("Transaction Amount Statistics — Fraud vs Genuine",
                 fontsize=14, color="white")
    ax.set_ylabel("Amount ($)", fontsize=12)
    ax.legend(fontsize=11)

    path = os.path.join(PLOTS_DIR, "fraud_amount_stats.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved fraud amount stats plot → {path}")
    return path


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> List[str]:
    """Execute every EDA plot and return the list of saved file paths."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    paths = [
        plot_class_distribution(df),
        plot_correlation_heatmap(df),
        plot_amount_distribution(df),
        plot_time_distribution(df),
        plot_pca_boxplots(df),
        plot_feature_importance(X, y),
        plot_fraud_amount_stats(df),     # NEW
    ]
    print(f"[eda] All EDA plots saved ({len(paths)} files).\n")
    return paths