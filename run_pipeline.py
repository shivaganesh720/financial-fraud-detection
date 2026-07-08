#!/usr/bin/env python
"""
run_pipeline.py — Entry point for the credit-card fraud detection pipeline.

Usage
-----
    python run_pipeline.py

Make sure ``creditcard.csv`` is in the same directory as this script.
Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
