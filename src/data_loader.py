"""
data_loader.py — Load and inspect the credit-card fraud dataset.

Expects ``creditcard.csv`` to live in the project root (path set in config).
"""

from __future__ import annotations

import sys
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import DATA_PATH


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the credit-card CSV and return the raw DataFrame.

    Parameters
    ----------
    path : str
        Absolute or relative path to ``creditcard.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at *path*.
    """
    import os

    if not os.path.isfile(path):
        print(
            f"\n[ERROR] Dataset not found at:\n  {path}\n"
            "Please download 'creditcard.csv' from Kaggle and place it in the "
            "project root directory.\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        )
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[data_loader] Loaded dataset from {path}")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print summary statistics about the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The raw credit-card dataset.
    """
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape           : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage    : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"\nColumn dtypes:\n{df.dtypes.value_counts().to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    if df.isnull().sum().sum() == 0:
        print("  (no nulls found)")
    print(f"\nClass distribution:")
    counts = df["Class"].value_counts()
    for cls, cnt in counts.items():
        pct = cnt / len(df) * 100
        label = "Genuine" if cls == 0 else "Fraud"
        print(f"  {label} (Class {cls}): {cnt:>7,}  ({pct:.3f}%)")
    print("=" * 60 + "\n")


def get_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset containing a ``Class`` column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) where X has all columns except ``Class``.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]
    print(f"[data_loader] Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y
