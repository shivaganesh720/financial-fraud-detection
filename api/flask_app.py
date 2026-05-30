"""
Flask REST API for Credit Card Fraud Detection — SentinelShield AI.

Endpoints
---------
POST /predict           – Classify a single transaction.
POST /predict/batch     – Classify multiple transactions in one call.
GET  /health            – Health-check (model loaded, MongoDB status).
GET  /model-info        – Model metadata and training metrics.
GET  /statistics        – Aggregate fraud statistics from MongoDB.
GET  /history           – Recent prediction history (newest first).
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger("fraud_api")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = MODELS_DIR / "training_metrics.json"

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
V_FEATURES = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES = ["Time", "Amount"] + V_FEATURES  # 30 features total

# ---------------------------------------------------------------------------
# MongoDB helper
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from mongo_utils import (
        save_prediction,
        is_connected as mongo_connected,
        get_prediction_history,
        get_fraud_statistics,
        get_risk_breakdown,
    )
except ImportError:
    logger.warning("mongo_utils not importable – MongoDB logging disabled.")

    def save_prediction(*_a, **_kw):
        return None

    def mongo_connected():
        return False

    def get_prediction_history(limit=50):
        return []

    def get_fraud_statistics():
        return {"total_predictions": 0, "total_frauds": 0,
                "total_legitimate": 0, "fraud_rate": 0.0}

    def get_risk_breakdown():
        return {"high": 0, "medium": 0, "low": 0}


# ---------------------------------------------------------------------------
# Model + scaler — loaded once at startup
# ---------------------------------------------------------------------------
def _load_artifacts() -> tuple[Any, Any, str | None]:
    """Load model and scaler from disk. Returns (model, scaler, error_msg)."""
    model, scaler, error = None, None, None
    try:
        if not BEST_MODEL_PATH.exists():
            error = f"Model file not found: {BEST_MODEL_PATH}"
            logger.error(error)
            return model, scaler, error
        model = joblib.load(BEST_MODEL_PATH)
        logger.info("Model loaded from %s  [type=%s]", BEST_MODEL_PATH, type(model).__name__)
    except Exception as exc:
        error = str(exc)
        logger.error("Failed to load model: %s", exc)
        return model, scaler, error

    try:
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded from %s", SCALER_PATH)
        else:
            logger.warning("Scaler not found at %s – raw Amount values will be used.", SCALER_PATH)
    except Exception as exc:
        logger.warning("Failed to load scaler: %s – raw Amount values will be used.", exc)

    return model, scaler, error


_MODEL, _SCALER, _LOAD_ERROR = _load_artifacts()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _classify_risk(probability: float) -> str:
    """Map a fraud probability to a human-readable risk tier."""
    if probability >= 0.75:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"


def _extract_features(data: dict) -> tuple[np.ndarray, str | None]:
    """
    Parse and validate feature values from a request payload.

    Returns
    -------
    (features_2d, error_message)
        features_2d is shaped (1, 29): V1–V28 + scaled Amount.
        error_message is None on success.
    """
    try:
        v_vals = [float(data.get(f, 0.0)) for f in V_FEATURES]
        amount = float(data.get("Amount", 0.0))
    except (TypeError, ValueError) as exc:
        return None, f"Invalid feature value: {exc}"

    if _SCALER is not None:
        try:
            scaled_amount = float(_SCALER.transform([[amount]])[0][0])
        except Exception as exc:
            logger.warning("Scaler transform failed, using raw Amount: %s", exc)
            scaled_amount = amount
    else:
        scaled_amount = amount

    features_2d = np.array(v_vals + [scaled_amount]).reshape(1, -1)
    return features_2d, None


def _run_inference(features_2d: np.ndarray) -> tuple[int, float, str | None]:
    """
    Run model prediction on a pre-built feature array.

    Returns
    -------
    (prediction, probability, error_message)
    """
    try:
        prediction = int(_MODEL.predict(features_2d)[0])
        if hasattr(_MODEL, "predict_proba"):
            probability = float(_MODEL.predict_proba(features_2d)[0][1])
        elif hasattr(_MODEL, "decision_function"):
            raw = float(_MODEL.decision_function(features_2d)[0])
            probability = 1.0 / (1.0 + np.exp(-raw))
        else:
            probability = float(prediction)
        return prediction, probability, None
    except Exception as exc:
        logger.exception("Prediction error")
        return 0, 0.0, f"Prediction failed: {exc}"


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


# ── Health check ────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": _MODEL is not None,
        "model_type": type(_MODEL).__name__ if _MODEL else None,
        "scaler_loaded": _SCALER is not None,
        "mongodb_connected": mongo_connected(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── Model info ──────────────────────────────────────────────────────────────

@app.route("/model-info", methods=["GET"])
def model_info():
    info: dict = {
        "model_loaded": _MODEL is not None,
        "model_type": type(_MODEL).__name__ if _MODEL else None,
        "scaler_loaded": _SCALER is not None,
        "features": ALL_FEATURES,
        "feature_count": len(ALL_FEATURES),
    }
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH) as fh:
                info["training_metrics"] = json.load(fh)
        except Exception:
            pass
    if _LOAD_ERROR:
        info["load_error"] = _LOAD_ERROR
    return jsonify(info)


# ── Single prediction ───────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    if _MODEL is None:
        return jsonify({"error": "Model not loaded", "detail": _LOAD_ERROR}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    features_2d, err = _extract_features(data)
    if err:
        return jsonify({"error": err}), 400

    prediction, probability, err = _run_inference(features_2d)
    if err:
        return jsonify({"error": err}), 500

    risk_level = _classify_risk(probability)
    mongo_id = save_prediction(
        transaction_data=data,
        prediction=prediction,
        probability=probability,
        risk_level=risk_level,
    )

    return jsonify({
        "fraud_probability": round(probability, 6),
        "is_fraud": bool(prediction),
        "risk_level": risk_level,
        "mongo_logged": mongo_id is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── Batch prediction ────────────────────────────────────────────────────────

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Classify a list of transactions in a single request.

    Request body: {"transactions": [ {transaction_1}, {transaction_2}, ... ]}
    Max batch size: 500 transactions.
    """
    if _MODEL is None:
        return jsonify({"error": "Model not loaded", "detail": _LOAD_ERROR}), 503

    data = request.get_json(silent=True)
    if data is None or "transactions" not in data:
        return jsonify({"error": "Body must be JSON with a 'transactions' list."}), 400

    transactions = data["transactions"]
    if not isinstance(transactions, list) or len(transactions) == 0:
        return jsonify({"error": "'transactions' must be a non-empty list."}), 400
    if len(transactions) > 500:
        return jsonify({"error": "Batch size exceeds maximum of 500."}), 400

    results = []
    errors = []

    for idx, tx in enumerate(transactions):
        features_2d, err = _extract_features(tx)
        if err:
            errors.append({"index": idx, "error": err})
            results.append(None)
            continue

        prediction, probability, err = _run_inference(features_2d)
        if err:
            errors.append({"index": idx, "error": err})
            results.append(None)
            continue

        risk_level = _classify_risk(probability)
        save_prediction(
            transaction_data=tx,
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
        )
        results.append({
            "index": idx,
            "fraud_probability": round(probability, 6),
            "is_fraud": bool(prediction),
            "risk_level": risk_level,
        })

    return jsonify({
        "total": len(transactions),
        "processed": len([r for r in results if r is not None]),
        "errors": errors,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── Statistics ──────────────────────────────────────────────────────────────

@app.route("/statistics", methods=["GET"])
def statistics():
    """Return aggregate fraud statistics and risk tier breakdown from MongoDB."""
    stats = get_fraud_statistics()
    breakdown = get_risk_breakdown()
    return jsonify({
        **stats,
        "risk_breakdown": breakdown,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── Prediction history ──────────────────────────────────────────────────────

@app.route("/history", methods=["GET"])
def history():
    """
    Return recent predictions, newest first.

    Query params:
        limit (int, default 50, max 500)
    """
    try:
        limit = min(int(request.args.get("limit", 50)), 500)
    except ValueError:
        return jsonify({"error": "'limit' must be an integer."}), 400

    records = get_prediction_history(limit=limit)

    # Serialise datetime objects to ISO strings for JSON
    for rec in records:
        if "timestamp" in rec and hasattr(rec["timestamp"], "isoformat"):
            rec["timestamp"] = rec["timestamp"].isoformat()

    return jsonify({
        "count": len(records),
        "limit": limit,
        "records": records,
    })


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    logger.info("Starting SentinelShield API on port %d  debug=%s", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)