from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone
import json
import statistics


LOG_PATH = Path("logs/inference.jsonl")
OUTPUT_PATH = Path("artifacts/shadow/analysis.json")


# ============================================================
# Utilities
# ============================================================

def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ============================================================
# Load & filter logs
# ============================================================

def load_events(
    *,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> List[Dict]:

    if not LOG_PATH.exists():
        raise FileNotFoundError("Inference log file not found.")

    events: List[Dict] = []

    with open(LOG_PATH, "r") as f:
        for line in f:
            record = json.loads(line)

            ts = _parse_ts(record["timestamp_utc"])

            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue

            events.append(record)

    return events


# ============================================================
# Pair prod + shadow requests
# ============================================================

def pair_requests(events: List[Dict]) -> List[Dict]:

    by_request: Dict[str, List[Dict]] = {}

    for e in events:
        rid = e["request_id"]
        by_request.setdefault(rid, []).append(e)

    paired: List[Dict] = []

    for rid, records in by_request.items():
        prod = next((r for r in records if r["status"] == "success"), None)
        shadow = next((r for r in records if r["status"] == "shadow"), None)

        if prod and shadow:
            paired.append(
                {
                    "request_id": rid,
                    "prod": prod,
                    "shadow": shadow,
                }
            )

    return paired


# ============================================================
# Metric computation
# ============================================================

def compute_prediction_deltas(pairs: List[Dict]) -> Dict:

    deltas = [
        pair["shadow"]["prediction"]["predicted_probability"]
        - pair["prod"]["prediction"]["predicted_probability"]
        for pair in pairs
    ]

    abs_deltas = [abs(d) for d in deltas]

    return {
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "p95_abs_delta": statistics.quantiles(abs_deltas, n=20)[18],
        "outlier_pct": sum(d > 0.05 for d in abs_deltas) / len(abs_deltas) * 100,
    }


def compute_latency_deltas(pairs: List[Dict]) -> Dict:

    deltas = [
        pair["shadow"]["latency_ms"] - pair["prod"]["latency_ms"]
        for pair in pairs
    ]

    shadow_latencies = [pair["shadow"]["latency_ms"] for pair in pairs]

    return {
        "mean_delta_ms": statistics.mean(deltas),
        "p95_shadow_ms": statistics.quantiles(shadow_latencies, n=20)[18],
    }


def compute_error_rates(pairs: List[Dict]) -> Dict:

    prod_errors = sum(pair["prod"]["error"] is not None for pair in pairs)
    shadow_errors = sum(pair["shadow"]["error"] is not None for pair in pairs)

    n = len(pairs)

    return {
        "production": prod_errors / n,
        "shadow": shadow_errors / n,
    }


# ============================================================
# Decision logic
# ============================================================

def make_decision(
    *,
    prediction_metrics: Dict,
    latency_metrics: Dict,
    error_rates: Dict,
) -> Dict:

    reasons: List[str] = []
    severity = "low"

    if prediction_metrics["p95_abs_delta"] > 0.1:
        reasons.append("High prediction divergence (p95 > 0.1)")
        severity = "high"

    if latency_metrics["mean_delta_ms"] > 20:
        reasons.append("Shadow latency regression > 20ms")
        severity = "medium"

    if error_rates["shadow"] > error_rates["production"]:
        reasons.append("Shadow error rate exceeds production")
        severity = "high"

    if reasons:
        return {
            "recommendation": "HOLD",
            "severity": severity,
            "reasons": reasons,
        }

    return {
        "recommendation": "SAFE_TO_PROMOTE",
        "severity": "low",
        "reasons": [],
    }


# ============================================================
# Main analysis entry point
# ============================================================

def run_shadow_analysis(
    *,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> Dict:

    events = load_events(start_time=start_time, end_time=end_time)
    pairs = pair_requests(events)

    if not pairs:
        raise RuntimeError("No paired shadow/production requests found.")

    prediction_metrics = compute_prediction_deltas(pairs)
    latency_metrics = compute_latency_deltas(pairs)
    error_rates = compute_error_rates(pairs)

    decision = make_decision(
        prediction_metrics=prediction_metrics,
        latency_metrics=latency_metrics,
        error_rates=error_rates,
    )

    artifact = {
        "analysis_window": {
            "start": start_time.isoformat() if start_time else None,
            "end": end_time.isoformat() if end_time else None,
        },
        "counts": {
            "paired_requests": len(pairs)
        },
        "prediction_comparison": prediction_metrics,
        "latency_comparison": latency_metrics,
        "error_rates": error_rates,
        "decision": decision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    return artifact