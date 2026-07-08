#!/usr/bin/env python
"""
generate_mock_csv.py — Generate a mock creditcard.csv for immediate local testing.

Creates a synthetic version of the Kaggle Credit Card Fraud dataset with
2,000 genuine transactions and 20 fraud transactions to allow testing the
entire ML pipeline immediately without requiring a large 150MB file download.
"""

import os
import random
import pandas as pd
import numpy as np

def generate_data():
    print("[*] Generating mock credit card dataset...")
    np.random.seed(42)
    random.seed(42)
    
    n_genuine = 2000
    n_fraud = 20
    
    v_features = [f"V{i}" for i in range(1, 29)]
    columns = ["Time", "Amount"] + v_features + ["Class"]
    
    rows = []
    
    # Generate genuine transactions (normal distribution centered around 0)
    for i in range(n_genuine):
        time_val = i * 40.0
        amount = round(random.lognormvariate(3.5, 1.2), 2)
        v_vals = np.random.normal(0, 1, 28).tolist()
        rows.append([time_val, amount] + v_vals + [0])
        
    # Generate fraud transactions (shifted distribution, higher variance, larger amounts)
    for i in range(n_fraud):
        time_val = random.uniform(0, n_genuine * 40.0)
        amount = round(random.uniform(500.0, 3000.0), 2)
        # Shift V-features typical of fraud patterns
        v_vals = np.random.normal(-1.5, 2.5, 28).tolist()
        # Ensure key variables are strongly negative for SHAP/models to learn patterns
        v_vals[2] = -4.0  # V3
        v_vals[11] = -5.0 # V12
        v_vals[13] = -6.5 # V14
        v_vals[16] = -7.0 # V17
        rows.append([time_val, amount] + v_vals + [1])
        
    df = pd.DataFrame(rows, columns=columns)
    
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    out_path = "creditcard.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Generated mock dataset with {len(df)} rows (Genuine: {n_genuine}, Fraud: {n_fraud}) at:\n  {os.path.abspath(out_path)}\n")

if __name__ == "__main__":
    generate_data()
