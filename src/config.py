"""
config.py — Project configuration constants.

Centralizes all paths, hyperparameters, and settings so every module
draws from a single source of truth.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH: str = os.path.join(PROJECT_ROOT, "creditcard.csv")

MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR: str = os.path.join(OUTPUTS_DIR, "plots")

BEST_MODEL_PATH: str = os.path.join(MODELS_DIR, "best_model.joblib")
SCALER_PATH: str = os.path.join(MODELS_DIR, "scaler.joblib")
MODEL_COMPARISON_PATH: str = os.path.join(OUTPUTS_DIR, "model_comparison.csv")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2

# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------
SMOTE_SAMPLING_STRATEGY: float = 0.5  # ratio of minority to majority after SMOTE

# ---------------------------------------------------------------------------
# Model hyper-parameters (defaults used before tuning)
# ---------------------------------------------------------------------------
LOGISTIC_REGRESSION_PARAMS: dict = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGBOOST_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1,  # will be overridden dynamically
    "random_state": RANDOM_STATE,
    "eval_metric": "aucpr",
    "use_label_encoder": False,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# Randomized search budget
# ---------------------------------------------------------------------------
N_ITER_RANDOM_SEARCH: int = 20
CV_FOLDS: int = 3

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
PLOT_STYLE: str = "dark_background"
ACCENT_COLOR: str = "#00CED1"       # dark turquoise / teal
ACCENT_COLOR_2: str = "#00FFFF"     # cyan
ACCENT_COLOR_3: str = "#FF6F61"     # coral (for contrast)
PLOT_DPI: int = 150
