import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone


# ============================================================
# Helpers
# ============================================================

def _parse_timestamp(ts: str) -> Optional[datetime]:
    try:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        return None


# ============================================================
# Core log reader
# ============================================================

def read_inference_logs(
    log_path: Path,
    *,
    since_minutes: Optional[int] = None,
) -> List[Dict[str, Any]]:

    if not log_path.exists():
        raise FileNotFoundError(f"Inference log file not found: {log_path}")

    now = datetime.now(timezone.utc)
    cutoff_time: Optional[datetime] = None

    if since_minutes is not None:
        cutoff_time = now - timedelta(minutes=since_minutes)

    records: List[Dict[str, Any]] = []

    with log_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed JSON lines
                continue

            # Basic sanity check
            if not isinstance(record, dict):
                continue

            # Timestamp filtering (if enabled)
            if cutoff_time is not None:
                ts = record.get("timestamp_utc") or record.get("timestamp")
                parsed_ts = _parse_timestamp(ts) if ts else None

                if parsed_ts is None:
                    continue

                if parsed_ts < cutoff_time:
                    continue

            records.append(record)

    return records


from typing import List, Dict, Any
import math


# ============================================================
# Metric helpers
# ============================================================

def _percentile(values: List[float], p: float) -> float:
    if not values:
        return math.nan

    values = sorted(values)
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return values[int(k)]

    return values[f] * (c - k) + values[c] * (k - f)


# ============================================================
# Core aggregation
# ============================================================

def aggregate_inference_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:

    total = len(records)

    status_counts = {
        "success": 0,
        "client_error": 0,
        "server_error": 0,
    }

    latencies = []
    inference_latencies = []
    predictions = []

    for r in records:
        status = r.get("status")
        if status in status_counts:
            status_counts[status] += 1

        # Latency
        latency = r.get("latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(latency)

        inference_ms = r.get("inference_ms")
        if isinstance(inference_ms, (int, float)):
            inference_latencies.append(inference_ms)

        # Prediction
        pred = (
            r.get("prediction", {}) or {}
        ).get("predicted_probability")

        if isinstance(pred, (int, float)):
            predictions.append(pred)

    success = status_counts["success"]
    error_total = (
        status_counts["client_error"]
        + status_counts["server_error"]
    )

    return {
        "volume": {
            "total_requests": total,
            "success": success,
            "client_error": status_counts["client_error"],
            "server_error": status_counts["server_error"],
            "error_rate": (error_total / total) if total else 0.0,
        },
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else math.nan,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        },
        "inference_ms": {
            "mean": sum(inference_latencies) / len(inference_latencies)
            if inference_latencies
            else math.nan,
            "p95": _percentile(inference_latencies, 0.95),
        },
        "predictions": {
            "count": len(predictions),
            "mean": sum(predictions) / len(predictions)
            if predictions
            else math.nan,
            "min": min(predictions) if predictions else math.nan,
            "max": max(predictions) if predictions else math.nan,
        },
    }

from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import List, Dict, Any


def bucket_records_by_time(
    records: List[Dict[str, Any]],
    window_minutes: int,
) -> Dict[datetime, List[Dict[str, Any]]]:

    buckets = defaultdict(list)

    for r in records:
        ts = r.get("timestamp_utc") or r.get("timestamp")
        if not ts:
            continue

        try:
            if ts.endswith("Z"):
                ts = ts.replace("Z", "+00:00")
            t = datetime.fromisoformat(ts)
        except Exception:
            continue

        window_start = t.replace(
            second=0,
            microsecond=0,
        ) - timedelta(
            minutes=t.minute % window_minutes
        )

        buckets[window_start].append(r)

    return dict(buckets)

def aggregate_metrics_by_window(
    records: List[Dict[str, Any]],
    window_minutes: int = 5,
) -> List[Dict[str, Any]]:

    buckets = bucket_records_by_time(records, window_minutes)
    results = []

    for window_start, window_records in sorted(buckets.items()):
        metrics = aggregate_inference_metrics(window_records)

        results.append({
            "window_start_utc": window_start.isoformat() + "Z",
            "window_minutes": window_minutes,
            "metrics": metrics,
        })

    return results
