from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


# ============================================================
# Core sub-schemas
# ============================================================

@dataclass(frozen=True)
class PredictionLog:
    predicted_probability: float


@dataclass(frozen=True)
class InputSummary:
    n_features_received: int
    n_features_expected: int
    missing_features: List[str]
    extra_features: List[str]
    feature_hash: str


@dataclass(frozen=True)
class FeatureStats:
    numeric: Dict[str, float]
    categorical: Dict[str, Any]


@dataclass(frozen=True)
class ErrorLog:
    error_type: str
    message: str


# ============================================================
# Top-level inference log event
# ============================================================

@dataclass(frozen=True)
class InferenceLogEvent:
    event_type: str
    schema_version: str

    request_id: str
    timestamp_utc: str

    model_name: str
    model_version: str

    status: str

    latency_ms: float
    inference_ms: Optional[float]

    prediction: Optional[PredictionLog]
    input_summary: InputSummary
    feature_stats: FeatureStats

    error: Optional[ErrorLog]

    # --------------------------------------------------------
    # Serialization helper
    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Helpers
# ============================================================

def utc_now_iso() -> str:
    """
    Return current UTC time as ISO-8601 string.
    """
    return datetime.utcnow().isoformat() + "Z"