"""
pipeline.py — Orchestrate the full fraud-detection ML pipeline.

Stages
------
1. Load data
2. Exploratory Data Analysis (EDA)
3. Preprocessing (scaling, splitting, SMOTE)
4. Train all models & compare
5. Select best model → hyperparameter tuning
6. Evaluate best model
7. SHAP explainability
8. Summary
"""

from __future__ import annotations

import os
import time

from src.config import MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR
from src.data_loader import load_data, inspect_data, get_features_target
from src.eda import run_eda
from src.preprocessing import preprocess
from src.train import train_all_models, select_best_model, tune_best_model
from src.evaluate import run_evaluation
from src.explainability import run_explainability


def run_full_pipeline() -> None:
    """Execute every stage of the fraud-detection pipeline end-to-end.

    Outputs are written to:
    - ``outputs/plots/``   — EDA and evaluation figures
    - ``outputs/``         — model comparison CSV, SHAP force-plot HTML
    - ``models/``          — fitted scaler + all model artefacts
    """
    start = time.time()
    print("╔" + "═" * 58 + "╗")
    print("║   CREDIT CARD FRAUD DETECTION — ML PIPELINE              ║")
    print("╚" + "═" * 58 + "╝")

    # Ensure output dirs
    for d in [MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ 1 --
    print("\n▸ Stage 1 / 7 — Loading Data")
    df = load_data()
    inspect_data(df)
    X, y = get_features_target(df)

    # ------------------------------------------------------------------ 2 --
    print("\n▸ Stage 2 / 7 — Exploratory Data Analysis")
    run_eda(df, X, y)

    # ------------------------------------------------------------------ 3 --
    print("\n▸ Stage 3 / 7 — Preprocessing")
    X_train_sm, X_test, y_train_sm, y_test, X_train_orig, y_train_orig = preprocess(X, y)

    # ------------------------------------------------------------------ 4 --
    print("\n▸ Stage 4 / 7 — Training Models")
    fitted_models, comparison_df = train_all_models(
        X_train_sm, y_train_sm, X_test, y_test,
    )

    # ------------------------------------------------------------------ 5 --
    print("\n▸ Stage 5 / 7 — Selecting & Tuning Best Model")
    best_name, best_model = select_best_model(fitted_models, comparison_df)
    tuned_model, tuned_metrics = tune_best_model(
        best_name, best_model, X_train_sm, y_train_sm, X_test, y_test,
    )
    # Update reference to tuned model
    fitted_models[f"{best_name} (tuned)"] = tuned_model

    # ------------------------------------------------------------------ 6 --
    print("\n▸ Stage 6 / 7 — Evaluating Models")
    run_evaluation(fitted_models, best_name, tuned_model, X_test, y_test)

    # ------------------------------------------------------------------ 7 --
    print("\n▸ Stage 7 / 7 — Explainability (SHAP)")
    run_explainability(tuned_model, X_test, model_name=best_name)

    # ---------------------------------------------------------------- done --
    elapsed = time.time() - start
    print("╔" + "═" * 58 + "╗")
    print("║   PIPELINE COMPLETE                                      ║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Total time : {elapsed:.1f}s")
    print(f"  Models     : {MODELS_DIR}")
    print(f"  Plots      : {PLOTS_DIR}")
    print(f"  Outputs    : {OUTPUTS_DIR}")
    print()
