from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import json
import uuid
import numpy as np


# ============================================================
# Snapshot builder
# ============================================================

def build_snapshot(
    records: List[Dict[str, Any]],
    *,
    model_name: str,
    model_version: str,
    feature_version: str,
    window_start: datetime,
    window_end: datetime,
) -> Dict[str, Any]:

    snapshot_id = str(uuid.uuid4())

    # -----------------------------
    # Volume stats
    # -----------------------------
    n_requests = len(records)
    n_success = sum(1 for r in records if r.get("status") == "success")
    n_errors = n_requests - n_success

    # -----------------------------
    # Prediction stats
    # -----------------------------
    preds = [
        r["prediction"]["predicted_probability"]
        for r in records
        if r.get("prediction") is not None
    ]

    prediction_stats = _numeric_distribution(preds) if preds else None

    # -----------------------------
    # Feature stats
    # -----------------------------
    numeric_features: Dict[str, List[float]] = {}
    categorical_features: Dict[str, Dict[str, int]] = {}

    for r in records:
        stats = r.get("feature_stats", {})
        numeric = stats.get("numeric", {})
        categorical = stats.get("categorical", {})

        for k, v in numeric.items():
            numeric_features.setdefault(k, []).append(v)

        for k, v in categorical.items():
            categorical_features.setdefault(k, {})
            categorical_features[k][v] = categorical_features[k].get(v, 0) + 1

    numeric_feature_stats = {
        name: _numeric_distribution(values)
        for name, values in numeric_features.items()
        if values
    }

    categorical_feature_stats = {
        name: _categorical_distribution(counts)
        for name, counts in categorical_features.items()
        if counts
    }

    # -----------------------------
    # Snapshot assembly
    # -----------------------------
    snapshot = {
        "schema_version": "1.0.0",
        "snapshot_id": snapshot_id,
        "model_name": model_name,
        "model_version": model_version,
        "feature_version": feature_version,
        "window": {
            "start_utc": window_start.isoformat() + "Z",
            "end_utc": window_end.isoformat() + "Z",
            "duration_minutes": int((window_end - window_start).total_seconds() / 60),
        },
        "volume": {
            "n_requests": n_requests,
            "n_success": n_success,
            "n_errors": n_errors,
        },
        "predictions": prediction_stats,
        "features": {
            "numeric": numeric_feature_stats,
            "categorical": categorical_feature_stats,
        },
    }

    return snapshot


# ============================================================
# Distribution helpers
# ============================================================

def _numeric_distribution(values: List[float]) -> Dict[str, Any]:
    arr = np.array(values, dtype=float)

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "quantiles": {
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        },
    }


def _categorical_distribution(counts: Dict[str, int]) -> Dict[str, Any]:
    total = sum(counts.values())
    frequencies = {
        k: v / total for k, v in counts.items()
    }

    return {
        "counts": counts,
        "frequencies": frequencies,
    }


# ============================================================
# Persistence helpers
# ============================================================

def save_snapshot(snapshot: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def load_snapshot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)