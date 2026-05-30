"""
MongoDB Utility Functions for SentinelShield AI.

Provides connection management, prediction logging, history retrieval,
fraud statistics, and risk-tier aggregations.

All functions gracefully handle MongoDB being unavailable — callers never
need to check for pymongo availability themselves.
"""

import os
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy pymongo import
# ---------------------------------------------------------------------------
try:
    from pymongo import MongoClient, DESCENDING, ASCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    logger.warning("pymongo not installed – MongoDB features disabled.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MONGO_URI        = os.getenv("MONGO_URI",        "mongodb://localhost:27017/")
MONGO_DB         = os.getenv("MONGO_DB",         "fraud_detection")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "predictions")
MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "2000"))

# ---------------------------------------------------------------------------
# Connection singleton
# ---------------------------------------------------------------------------
_client: Optional[Any] = None
_db:     Optional[Any] = None
_indexes_created = False


def _create_indexes(collection: Any) -> None:
    """Create useful indexes on first connection (idempotent)."""
    global _indexes_created
    if _indexes_created:
        return
    try:
        collection.create_index([("timestamp",  DESCENDING)], background=True)
        collection.create_index([("prediction", ASCENDING)],  background=True)
        collection.create_index([("risk_level", ASCENDING)],  background=True)
        _indexes_created = True
        logger.info("MongoDB indexes ensured on '%s'.", MONGO_COLLECTION)
    except Exception as exc:
        logger.warning("Could not create indexes: %s", exc)


def _get_db(retry: bool = True) -> Optional[Any]:
    """
    Return a MongoDB database handle, reconnecting once on failure.

    Returns None when MongoDB is genuinely unavailable.
    """
    global _client, _db
    if not PYMONGO_AVAILABLE:
        return None

    def _connect() -> Optional[Any]:
        global _client, _db
        try:
            _client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
                connectTimeoutMS=MONGO_TIMEOUT_MS,
            )
            _client.admin.command("ping")
            _db = _client[MONGO_DB]
            _create_indexes(_db[MONGO_COLLECTION])
            logger.info("Connected to MongoDB at %s", MONGO_URI)
            return _db
        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            logger.warning("MongoDB unavailable: %s", exc)
            _client = None
            _db = None
            return None
        except Exception as exc:
            logger.warning("Unexpected MongoDB error: %s", exc)
            _client = None
            _db = None
            return None

    if _client is None:
        return _connect()

    # Verify existing connection is still alive
    try:
        _client.admin.command("ping")
        return _db
    except Exception:
        logger.warning("MongoDB connection lost — reconnecting…")
        _client = None
        _db = None
        if retry:
            time.sleep(0.5)
            return _connect()
        return None


def is_connected() -> bool:
    """Return True if MongoDB is currently reachable."""
    return _get_db() is not None


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_prediction(
    transaction_data: Dict[str, Any],
    prediction: int,
    probability: float,
    risk_level: str,
) -> Optional[str]:
    """
    Persist a single prediction record to MongoDB.

    Returns the inserted document's ``_id`` as a string, or None on failure.
    """
    db = _get_db()
    if db is None:
        return None
    try:
        doc = {
            "transaction_data": transaction_data,
            "prediction":       int(prediction),
            "probability":      round(float(probability), 6),
            "risk_level":       risk_level,
            "timestamp":        datetime.now(timezone.utc),
        }
        result = db[MONGO_COLLECTION].insert_one(doc)
        logger.debug("Prediction saved – id=%s", result.inserted_id)
        return str(result.inserted_id)
    except Exception as exc:
        logger.error("Failed to save prediction: %s", exc)
        return None


def clear_history() -> int:
    """
    Delete all prediction records from the collection.

    Useful for resetting test data. Returns the number of deleted documents,
    or -1 if MongoDB is unavailable.
    """
    db = _get_db()
    if db is None:
        return -1
    try:
        result = db[MONGO_COLLECTION].delete_many({})
        logger.info("Cleared %d prediction records.", result.deleted_count)
        return result.deleted_count
    except Exception as exc:
        logger.error("Failed to clear history: %s", exc)
        return -1


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_prediction_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent *limit* predictions, newest first.

    Returns an empty list when MongoDB is unavailable.
    """
    db = _get_db()
    if db is None:
        return []
    try:
        cursor = (
            db[MONGO_COLLECTION]
            .find({}, {"_id": 0})
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        return list(cursor)
    except Exception as exc:
        logger.error("Failed to fetch prediction history: %s", exc)
        return []


def get_fraud_statistics() -> Dict[str, Any]:
    """
    Compute aggregate fraud statistics across all stored predictions.

    Returns a dict with:
        total_predictions, total_frauds, total_legitimate, fraud_rate,
        avg_probability
    All zeros when MongoDB is unavailable.
    """
    empty: Dict[str, Any] = {
        "total_predictions": 0,
        "total_frauds":      0,
        "total_legitimate":  0,
        "fraud_rate":        0.0,
        "avg_probability":   0.0,
    }
    db = _get_db()
    if db is None:
        return empty
    try:
        pipeline = [
            {
                "$group": {
                    "_id":             None,
                    "total":           {"$sum": 1},
                    "frauds":          {"$sum": "$prediction"},
                    "avg_probability": {"$avg": "$probability"},
                }
            }
        ]
        rows = list(db[MONGO_COLLECTION].aggregate(pipeline))
        if not rows:
            return empty

        row = rows[0]
        total      = int(row["total"])
        frauds     = int(row["frauds"])
        legitimate = total - frauds
        fraud_rate = (frauds / total * 100) if total > 0 else 0.0
        avg_prob   = round(float(row.get("avg_probability") or 0.0), 4)

        return {
            "total_predictions": total,
            "total_frauds":      frauds,
            "total_legitimate":  legitimate,
            "fraud_rate":        round(fraud_rate, 2),
            "avg_probability":   avg_prob,
        }
    except Exception as exc:
        logger.error("Failed to compute fraud statistics: %s", exc)
        return empty


def get_risk_breakdown() -> Dict[str, int]:
    """
    Count predictions per risk tier (high / medium / low).

    Returns a dict with keys 'high', 'medium', 'low'.
    All zeros when MongoDB is unavailable.
    """
    empty = {"high": 0, "medium": 0, "low": 0}
    db = _get_db()
    if db is None:
        return empty
    try:
        pipeline = [
            {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}
        ]
        rows = list(db[MONGO_COLLECTION].aggregate(pipeline))
        breakdown = {"high": 0, "medium": 0, "low": 0}
        for row in rows:
            tier = row.get("_id", "")
            if tier in breakdown:
                breakdown[tier] = int(row["count"])
        return breakdown
    except Exception as exc:
        logger.error("Failed to compute risk breakdown: %s", exc)
        return empty