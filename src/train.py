"""
train.py — Train, evaluate, compare, and tune classification models.

Three models are trained (Logistic Regression, Random Forest, XGBoost).
The best model by AUPRC is selected, tuned with RandomizedSearchCV, and
persisted alongside all others. Training metrics are exported to
``models/training_metrics.json`` for the /model-info API endpoint.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    LOGISTIC_REGRESSION_PARAMS,
    MODEL_COMPARISON_PATH,
    MODELS_DIR,
    N_ITER_RANDOM_SEARCH,
    OUTPUTS_DIR,
    RANDOM_FOREST_PARAMS,
    RANDOM_STATE,
    XGBOOST_PARAMS,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Hyperparameter search spaces (previously buried inside this module)
# Kept here for locality; config.py holds the base defaults.
# ---------------------------------------------------------------------------
_PARAM_DISTRIBUTIONS: Dict[str, Dict] = {
    "LogisticRegression": {
        "C":        loguniform(1e-3, 1e2),
        "solver":   ["lbfgs", "saga"],
        "max_iter": [1000, 2000],
    },
    "RandomForest": {
        "n_estimators":     randint(100, 500),
        "max_depth":        randint(6, 25),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
    },
    "XGBoost": {
        "n_estimators":    randint(100, 500),
        "max_depth":       randint(3, 12),
        "learning_rate":   loguniform(1e-3, 0.3),
        "subsample":       uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
    },
}


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "",
) -> Dict[str, Any]:
    """Return a standard classification metrics dict."""
    metrics: Dict[str, Any] = {
        "Accuracy":  round(accuracy_score(y_true, y_pred),                     4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0),   4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0),       4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0),           4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob),                      4),
        "AUPRC":     round(average_precision_score(y_true, y_prob),            4),
    }
    if model_name:
        metrics["model_name"] = model_name
    return metrics


# ---------------------------------------------------------------------------
# Train a single model
# ---------------------------------------------------------------------------

def _train_single(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, Any]]:
    """Fit *model* and return (fitted_model, metrics_dict)."""
    print(f"  Training {name} …")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = _compute_metrics(y_test, y_pred, y_prob, model_name=name)

    print(
        f"    AUPRC={metrics['AUPRC']:.4f}  ROC-AUC={metrics['ROC-AUC']:.4f}  "
        f"F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}"
    )
    return model, metrics


# ---------------------------------------------------------------------------
# Train all models
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train Logistic Regression, Random Forest, and XGBoost.

    Returns
    -------
    (fitted_models, comparison_df)
        fitted_models : dict of name → fitted estimator
        comparison_df : metrics DataFrame sorted by AUPRC descending
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Build XGBoost params — remove deprecated use_label_encoder,
    # set scale_pos_weight dynamically from the training class balance.
    xgb_params = {
        k: v for k, v in XGBOOST_PARAMS.items()
        if k != "use_label_encoder"       # removed in XGBoost 2.x
    }
    n_neg = int((y_train == 0).sum())
    n_pos = max(int((y_train == 1).sum()), 1)
    xgb_params["scale_pos_weight"] = n_neg / n_pos

    candidates: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        "RandomForest":       RandomForestClassifier(**RANDOM_FOREST_PARAMS),
        "XGBoost":            XGBClassifier(**xgb_params),
    }

    fitted_models: Dict[str, Any] = {}
    all_metrics:   List[Dict[str, Any]] = []

    for name, model in candidates.items():
        model, metrics = _train_single(
            name, model,
            X_train.values, y_train.values,
            X_test.values,  y_test.values,
        )
        fitted_models[name] = model
        all_metrics.append(metrics)

        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)

    comparison_df = (
        pd.DataFrame(all_metrics)
        .set_index("model_name")
        .sort_values("AUPRC", ascending=False)
    )

    comparison_df.to_csv(MODEL_COMPARISON_PATH)
    print(f"\n[train] Model comparison saved → {MODEL_COMPARISON_PATH}")
    print(comparison_df.to_string())

    return fitted_models, comparison_df


# ---------------------------------------------------------------------------
# Select best model
# ---------------------------------------------------------------------------

def select_best_model(
    fitted_models: Dict[str, Any],
    comparison_df: pd.DataFrame,
) -> Tuple[str, Any]:
    """
    Pick the highest-AUPRC model and persist it as ``best_model.joblib``.

    Returns (best_name, fitted_model).
    """
    best_name  = comparison_df["AUPRC"].idxmax()
    best_model = fitted_models[best_name]

    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\n[train] Best model: {best_name}  AUPRC={comparison_df.loc[best_name, 'AUPRC']:.4f}")
    print(f"        Saved → {BEST_MODEL_PATH}")
    return best_name, best_model


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_best_model(
    best_name: str,
    best_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run RandomizedSearchCV on the best model and re-evaluate on the test set.

    Also saves ``models/training_metrics.json`` so the /model-info endpoint
    can serve up-to-date metrics without re-reading the CSV.

    Returns (tuned_model, metrics_dict).
    """
    print(f"\n[train] Hyperparameter tuning — {best_name} …")

    param_dist = _PARAM_DISTRIBUTIONS.get(best_name)
    if param_dist is None:
        print(f"[train] No param distribution defined for {best_name}. Skipping tuning.")
        return best_model, {}

    search = RandomizedSearchCV(
        estimator=best_model,
        param_distributions=param_dist,
        n_iter=N_ITER_RANDOM_SEARCH,
        scoring="average_precision",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train.values, y_train.values)
    tuned_model = search.best_estimator_

    print(f"[train] Best params: {search.best_params_}")

    y_pred = tuned_model.predict(X_test.values)
    y_prob = tuned_model.predict_proba(X_test.values)[:, 1]
    metrics = _compute_metrics(y_test.values, y_pred, y_prob, model_name=best_name)

    # Persist tuned model in two locations
    tuned_path = os.path.join(MODELS_DIR, f"{best_name}_tuned.joblib")
    joblib.dump(tuned_model, tuned_path)
    joblib.dump(tuned_model, BEST_MODEL_PATH)
    print(f"[train] Tuned model saved → {tuned_path}")
    print(
        f"  AUPRC={metrics['AUPRC']:.4f}  ROC-AUC={metrics['ROC-AUC']:.4f}  "
        f"F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}"
    )

    # Export metrics JSON for the /model-info endpoint
    metrics_path = os.path.join(MODELS_DIR, "training_metrics.json")
    metrics_export = {
        "model_name":   best_name,
        "best_params":  search.best_params_,
        "test_metrics": metrics,
    }
    with open(metrics_path, "w") as fh:
        json.dump(metrics_export, fh, indent=2)
    print(f"[train] Training metrics exported → {metrics_path}")

    return tuned_model, metrics