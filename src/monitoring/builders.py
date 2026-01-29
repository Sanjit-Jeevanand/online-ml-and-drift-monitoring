import hashlib
import json
import uuid
from typing import Dict, Any, List, Optional

from src.monitoring.schema import (
    InferenceLogEvent,
    PredictionLog,
    InputSummary,
    FeatureStats,
    ErrorLog,
    utc_now_iso,
)


# ============================================================
# Config
# ============================================================

SCHEMA_VERSION = "1.0.0"


# ============================================================
# Helpers
# ============================================================

def _stable_feature_hash(features: Dict[str, Any]) -> str:

    payload = json.dumps(features, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_input_summary(
    raw_features: Dict[str, Any],
    expected_features: List[str],
) -> InputSummary:

    received = set(raw_features.keys())
    expected = set(expected_features)

    missing = sorted(list(expected - received))
    extra = sorted(list(received - expected))

    feature_hash = _stable_feature_hash(raw_features)

    return InputSummary(
        n_features_received=len(received),
        n_features_expected=len(expected),
        missing_features=missing,
        extra_features=extra,
        feature_hash=feature_hash,
    )


def _build_feature_stats(
    features: Dict[str, Any],
    max_numeric: int = 5,
    max_categorical: int = 5,
) -> FeatureStats:

    numeric_stats: Dict[str, float] = {}
    categorical_stats: Dict[str, Any] = {}

    for k, v in features.items():
        if isinstance(v, (int, float)):
            if len(numeric_stats) < max_numeric:
                numeric_stats[k] = float(v)
        else:
            if len(categorical_stats) < max_categorical:
                categorical_stats[k] = v

        if len(numeric_stats) >= max_numeric and len(categorical_stats) >= max_categorical:
            break

    return FeatureStats(
        numeric=numeric_stats,
        categorical=categorical_stats,
    )


def _build_error_log(error: Optional[Exception]) -> Optional[ErrorLog]:

    if error is None:
        return None

    return ErrorLog(
        error_type=type(error).__name__,
        message=str(error),
    )


# ============================================================
# Public factory
# ============================================================

def build_inference_log_event(
    *,
    model_name: str,
    model_version: str,
    raw_features: Dict[str, Any],
    expected_features: List[str],
    predicted_probability: Optional[float],
    latency_ms: float,
    inference_ms: Optional[float],
    status: str,
    request_id: str,
    error: Optional[Exception] = None,
) -> InferenceLogEvent:

    input_summary = _build_input_summary(
        raw_features=raw_features,
        expected_features=expected_features,
    )

    feature_stats = _build_feature_stats(raw_features)

    prediction_log = (
        PredictionLog(predicted_probability=predicted_probability)
        if predicted_probability is not None
        else None
    )

    error_log = _build_error_log(error)

    return InferenceLogEvent(
        event_type="inference",
        schema_version=SCHEMA_VERSION,

        request_id=request_id,
        timestamp_utc=utc_now_iso(),

        model_name=model_name,
        model_version=model_version,

        status=status,

        latency_ms=round(latency_ms, 3),
        inference_ms=round(inference_ms, 3) if inference_ms is not None else None,

        prediction=prediction_log,
        input_summary=input_summary,
        feature_stats=feature_stats,

        error=error_log,
    )