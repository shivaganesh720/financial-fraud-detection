#!/usr/bin/env python
"""
simulator.py — Real-time transaction stream simulator for SentinelShield AI.

Reads rows from ``creditcard.csv`` (or generates synthetic data) and posts
them at configurable intervals to the Flask REST API, simulating a live feed.

Usage
-----
    python api/simulator.py --interval 0.5 --api http://localhost:5000
    python api/simulator.py --interval 1.0 --fraud-rate 0.15 --count 200
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Synthetic transaction generator
# ---------------------------------------------------------------------------

def generate_mock_transaction(is_fraud: bool = False) -> dict:
    """Generate a realistic PCA-feature transaction payload."""
    loc = -1.8 if is_fraud else 0.0
    v_vals = np.random.normal(loc=loc, scale=1.3, size=28).tolist()
    if is_fraud:
        # Amplify the key features the model was trained to detect
        v_vals[2]  = -4.0   # V3
        v_vals[11] = -5.0   # V12
        v_vals[13] = -6.5   # V14
        v_vals[16] = -7.0   # V17
    payload = {
        "Time":   float(time.time()),
        "Amount": float(round(random.uniform(500.0, 2500.0) if is_fraud
                              else random.lognormvariate(3.5, 1.2), 2)),
    }
    for idx, v in enumerate(v_vals, start=1):
        payload[f"V{idx}"] = float(v)
    return payload


# ---------------------------------------------------------------------------
# API health check with retry
# ---------------------------------------------------------------------------

def _wait_for_api(api_url: str, retries: int = 5, delay: float = 2.0) -> bool:
    """
    Poll /health until the API responds 200 or retries are exhausted.

    Returns True if the API is reachable, False otherwise.
    Unlike the original version this does NOT exit the simulator on failure —
    it falls back to local console-only mode so streaming still works.
    """
    health_url = f"{api_url}/health"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                print("🟢 API service is online and healthy.")
                return True
        except requests.RequestException:
            pass
        if attempt < retries:
            print(f"[!] API not reachable (attempt {attempt}/{retries}). Retrying in {delay}s…")
            time.sleep(delay)

    print("❌ API service offline. Running in console-only simulation mode.")
    return False


# ---------------------------------------------------------------------------
# Rolling summary printer
# ---------------------------------------------------------------------------

def _print_summary(count: int, frauds: int, prob_sum: float) -> None:
    """Print a rolling statistics line every 50 transactions."""
    fraud_rate = (frauds / count * 100) if count > 0 else 0.0
    avg_prob   = (prob_sum / count) if count > 0 else 0.0
    print(
        f"\n  ── Rolling summary (last {count} tx) ──\n"
        f"     Fraud rate  : {fraud_rate:.1f}%  ({frauds}/{count})\n"
        f"     Avg prob    : {avg_prob:.4f}\n"
    )


# ---------------------------------------------------------------------------
# Main streaming loop
# ---------------------------------------------------------------------------

def run_simulator(
    interval: float,
    api_url: str,
    fraud_rate: float = 0.10,
    max_count: int | None = None,
) -> None:
    """
    Simulate a streaming credit card transaction pipeline.

    Parameters
    ----------
    interval : float
        Seconds between transaction posts.
    api_url : str
        Root URL of the Flask prediction API.
    fraud_rate : float
        Fraction of synthetic transactions that are injected as fraud (0–1).
    max_count : int | None
        Stop after this many transactions. None means run indefinitely.
    """
    print("╔" + "═" * 58 + "╗")
    print("║   SENTINELSHIELD AI — REAL-TIME TRANSACTION STREAMER     ║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Target API   : {api_url}/predict")
    print(f"  Interval     : {interval}s")
    print(f"  Fraud rate   : {fraud_rate*100:.0f}%  (synthetic mode)")
    print(f"  Max count    : {'∞' if max_count is None else max_count}")
    print("  Status       : Press Ctrl+C to stop.\n")

    # ── Load dataset ──────────────────────────────────────────────────────
    csv_path = PROJECT_ROOT / "creditcard.csv"
    use_dataset = False
    df_sample = None

    if csv_path.exists():
        try:
            print(f"[*] Loading transactions from {csv_path.name}…")
            df = pd.read_csv(csv_path)
            genuine = df[df["Class"] == 0].sample(frac=0.05, random_state=42)
            fraud   = df[df["Class"] == 1]
            df_sample = (
                pd.concat([genuine, fraud])
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
            )
            use_dataset = True
            print(f"[*] Loaded {len(df_sample):,} rows for streaming.")
        except Exception as exc:
            print(f"[!] Could not load dataset ({exc}). Falling back to synthetic mode.")
    else:
        print("[*] No creditcard.csv found — using high-fidelity synthetic generator.")

    # ── Wait for API ──────────────────────────────────────────────────────
    api_online = _wait_for_api(api_url)

    # ── Streaming loop ────────────────────────────────────────────────────
    count = 0
    frauds_flagged = 0
    prob_sum = 0.0

    try:
        while True:
            if max_count is not None and count >= max_count:
                print(f"\n⏹️  Reached --count limit of {max_count}. Stopping.")
                break

            # 1. Build payload
            if use_dataset and df_sample is not None:
                row = df_sample.iloc[count % len(df_sample)]
                payload: dict = {"Time": float(row["Time"]), "Amount": float(row["Amount"])}
                for i in range(1, 29):
                    payload[f"V{i}"] = float(row[f"V{i}"])
            else:
                is_fraud_tx = random.random() < fraud_rate
                payload = generate_mock_transaction(is_fraud=is_fraud_tx)

            timestamp  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            amount_fmt = f"${payload['Amount']:>8.2f}"

            # 2. Send to API (or simulate locally)
            if api_online:
                try:
                    res = requests.post(
                        f"{api_url}/predict", json=payload, timeout=1.5
                    )
                    if res.status_code == 200:
                        data          = res.json()
                        prob          = data.get("fraud_probability", 0.0)
                        is_fraud_pred = data.get("is_fraud", False)
                        risk          = data.get("risk_level", "low").upper()
                        prob_sum     += prob
                        if is_fraud_pred:
                            frauds_flagged += 1
                        badge = "🔴 [FRAUD]   " if is_fraud_pred else "🟢 [OK]     "
                        print(
                            f"[{timestamp}] #{count+1:<4} │ {amount_fmt} │ "
                            f"p={prob:.4f} │ {badge} │ {risk}"
                        )
                    else:
                        print(f"[{timestamp}] #{count+1:<4} │ {amount_fmt} │ "
                              f"⚠️  HTTP {res.status_code}")
                except requests.RequestException:
                    # API went offline mid-stream — switch to local mode
                    api_online = False
                    print(f"[{timestamp}] #{count+1:<4} │ {amount_fmt} │ "
                          "⚠️  API unreachable — switching to local mode")
            else:
                # Local console simulation (no real inference)
                is_anomaly = (
                    payload.get("V14", 0) + payload.get("V17", 0) < -7.0
                )
                prob_mock = (
                    random.uniform(0.75, 0.99)
                    if is_anomaly
                    else random.uniform(0.01, 0.25)
                )
                badge = "🔴 [FRAUD]   " if is_anomaly else "🟢 [OK]     "
                prob_sum += prob_mock
                if is_anomaly:
                    frauds_flagged += 1
                print(
                    f"[{timestamp}] #{count+1:<4} │ {amount_fmt} │ "
                    f"p={prob_mock:.4f} │ {badge} │ (LOCAL)"
                )

            count += 1

            # 3. Rolling summary every 50 transactions
            if count % 50 == 0:
                _print_summary(count, frauds_flagged, prob_sum)

            time.sleep(interval)

    except KeyboardInterrupt:
        pass

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n\n⏹️  Transaction stream stopped.")
    print(f"  Total processed : {count:,}")
    print(f"  Frauds flagged  : {frauds_flagged:,}")
    if count > 0:
        print(f"  Fraud rate      : {frauds_flagged/count*100:.1f}%")
        print(f"  Avg probability : {prob_sum/count:.4f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SentinelShield AI — Live Transaction Stream Simulator"
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Seconds between transaction posts (default: 0.5)",
    )
    parser.add_argument(
        "--api", type=str, default="http://localhost:5000",
        help="Root URL of the Flask predict API (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--fraud-rate", type=float, default=0.10, dest="fraud_rate",
        help="Fraction of synthetic transactions injected as fraud, 0–1 (default: 0.10)",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Stop after N transactions. Omit to run indefinitely.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.fraud_rate <= 1.0):
        parser.error("--fraud-rate must be between 0.0 and 1.0")

    run_simulator(
        interval=args.interval,
        api_url=args.api,
        fraud_rate=args.fraud_rate,
        max_count=args.count,
    )