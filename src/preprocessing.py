"""
preprocessing.py — Feature engineering, scaling, splitting, and SMOTE.

Key invariant: the scaler is fit on the training split only — no leakage.

Improvements vs original
-------------------------
- _validate_features(): warns loudly when any column has >5% nulls or an
  unexpected dtype so bad data never silently reaches the model.
- Optional Time-based feature engineering: extract hour_of_day and
  time_period (morning/afternoon/evening/night) before dropping raw Time.
- Preprocessing summary saved to outputs/preprocessing_summary.json so the
  pipeline run is fully reproducible and auditable.
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
    SCALER_PATH,
    SMOTE_SAMPLING_STRATEGY,
    TEST_SIZE,
)


# ---------------------------------------------------------------------------
# 0. Feature validation
# ---------------------------------------------------------------------------

def _validate_features(X: pd.DataFrame) -> None:
    """
    Warn about data-quality issues that can silently degrade model performance.

    Checks
    ------
    - Columns with > 5% null values.
    - Non-numeric columns (unexpected dtypes).
    - Completely constant columns (zero variance).
    """
    issues = []

    null_fracs = X.isnull().mean()
    high_null  = null_fracs[null_fracs > 0.05]
    if not high_null.empty:
        for col, frac in high_null.items():
            issues.append(f"  ⚠ '{col}' has {frac*100:.1f}% null values")

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        issues.append(f"  ⚠ Non-numeric columns found: {non_numeric}")

    zero_var = X.columns[X.std() == 0].tolist()
    if zero_var:
        issues.append(f"  ⚠ Zero-variance (constant) columns: {zero_var}")

    if issues:
        print("[preprocessing] ── Data quality warnings ──")
        for msg in issues:
            print(msg)
    else:
        print("[preprocessing] Data quality check passed — no issues found.")


# ---------------------------------------------------------------------------
# 1. Optional Time-based feature engineering
# ---------------------------------------------------------------------------

def engineer_time_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Derive hour_of_day and time_period from the raw Time column, then drop it.

    The Kaggle dataset's Time column is seconds elapsed from the first
    transaction in a two-day window (~172,800 s max).  Hour-of-day and
    time-of-day period may carry weak but useful temporal signal.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing a 'Time' column.

    Returns
    -------
    pd.DataFrame
        Feature matrix with 'Time' replaced by 'hour_of_day' and
        'time_period' (0=night, 1=morning, 2=afternoon, 3=evening).
    """
    X = X.copy()
    if "Time" not in X.columns:
        return X

    # Seconds → hour of day (0–23), cycling through two days
    X["hour_of_day"] = (X["Time"] // 3600 % 24).astype(int)

    def _period(h: int) -> int:
        if 6 <= h < 12:   return 1  # morning
        if 12 <= h < 18:  return 2  # afternoon
        if 18 <= h < 24:  return 3  # evening
        return 0                     # night (0–5)

    X["time_period"] = X["hour_of_day"].apply(_period)
    X = X.drop(columns=["Time"])
    print("[preprocessing] Engineered 'hour_of_day' and 'time_period' from Time.")
    return X


# ---------------------------------------------------------------------------
# 2. Drop raw Time (simple path — no engineering)
# ---------------------------------------------------------------------------

def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """Drop the raw Time column (use engineer_time_features for richer path)."""
    X = X.copy()
    if "Time" in X.columns:
        X = X.drop(columns=["Time"])
        print("[preprocessing] Dropped 'Time' column.")
    return X


# ---------------------------------------------------------------------------
# 3. Stratified split
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving the class imbalance in both sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE,
    )
    print(
        f"[preprocessing] Split — train: {X_train.shape[0]:,}  "
        f"test: {X_test.shape[0]:,}  (stratified)"
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. Fit scaler on train only, transform both sets
# ---------------------------------------------------------------------------

def fit_and_scale(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    columns_to_scale: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on training data only, then transform both splits.

    The fitted scaler is persisted to models/scaler.joblib.

    Parameters
    ----------
    columns_to_scale : list[str] | None
        Defaults to ['Amount'] when present in X_train.
    """
    if columns_to_scale is None:
        columns_to_scale = ["Amount"] if "Amount" in X_train.columns else []

    scaler  = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()

    if columns_to_scale:
        X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
        X_test[columns_to_scale]  = scaler.transform(X_test[columns_to_scale])
        print(f"[preprocessing] Scaled {columns_to_scale} — fit on train only.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[preprocessing] Scaler saved → {SCALER_PATH}")

    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# 5. SMOTE (training set only)
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = SMOTE_SAMPLING_STRATEGY,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to the training set only — never to the test set."""
    print("[preprocessing] Applying SMOTE to training data…")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    print(
        f"[preprocessing] SMOTE complete — "
        f"before: {len(y_train):,}  after: {len(y_sm):,}  "
        f"(fraud: {(y_sm==1).sum():,}  genuine: {(y_sm==0).sum():,})"
    )
    return (
        pd.DataFrame(X_sm, columns=X_train.columns),
        pd.Series(y_sm, name="Class"),
    )


# ---------------------------------------------------------------------------
# 6. Preprocessing summary export
# ---------------------------------------------------------------------------

def _save_summary(
    original_shape:    tuple,
    train_shape:       tuple,
    test_shape:        tuple,
    smote_shape:       tuple,
    fraud_train_orig:  int,
    fraud_train_smote: int,
    fraud_test:        int,
) -> None:
    """Write a preprocessing summary JSON to outputs/ for auditability."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    summary = {
        "original_rows":         original_shape[0],
        "original_features":     original_shape[1],
        "train_rows_pre_smote":  train_shape[0],
        "train_rows_post_smote": smote_shape[0],
        "test_rows":             test_shape[0],
        "fraud_in_train_orig":   fraud_train_orig,
        "fraud_in_train_smote":  fraud_train_smote,
        "fraud_in_test":         fraud_test,
    }
    path = os.path.join(OUTPUTS_DIR, "preprocessing_summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[preprocessing] Summary saved → {path}")


# ---------------------------------------------------------------------------
# 7. Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    engineer_time: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature matrix (includes Time and Amount).
    y : pd.Series
        Binary target (0 = genuine, 1 = fraud).
    engineer_time : bool
        If True, derive hour_of_day / time_period from Time before dropping it.
        If False (default), simply drop Time.

    Returns
    -------
    (X_train_sm, X_test, y_train_sm, y_test, X_train_orig, y_train_orig)
        The *_orig variants are pre-SMOTE and useful for calibration.
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    original_shape = X.shape

    # 0. Validate
    _validate_features(X)

    # 1. Time handling
    if engineer_time:
        X = engineer_time_features(X)
    else:
        X = scale_features(X)

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Scale Amount
    X_train, X_test, _ = fit_and_scale(X_train, X_test)

    # 4. Preserve pre-SMOTE copies
    X_train_orig = X_train.copy()
    y_train_orig = y_train.copy()

    # 5. SMOTE
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # 6. Save summary
    _save_summary(
        original_shape    = original_shape,
        train_shape       = X_train.shape,
        test_shape        = X_test.shape,
        smote_shape       = X_train_sm.shape,
        fraud_train_orig  = int((y_train == 1).sum()),
        fraud_train_smote = int((y_train_sm == 1).sum()),
        fraud_test        = int((y_test == 1).sum()),
    )

    print("[preprocessing] Preprocessing complete.\n")
    return X_train_sm, X_test, y_train_sm, y_test, X_train_orig, y_train_orig