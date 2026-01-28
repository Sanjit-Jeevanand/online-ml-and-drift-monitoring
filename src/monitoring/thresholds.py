from typing import Dict, Literal


# ============================================================
# Threshold levels
# ============================================================

DriftLevel = Literal["none", "low", "medium", "high"]


# ============================================================
# Numeric feature drift (KS)
# ============================================================

KS_THRESHOLDS = {
    "low": 0.10,
    "medium": 0.20,
}


def classify_ks(ks_value: float) -> DriftLevel:
    if ks_value < KS_THRESHOLDS["low"]:
        return "none"
    if ks_value < KS_THRESHOLDS["medium"]:
        return "low"
    return "high"


# ============================================================
# Categorical feature drift (PSI)
# ============================================================

PSI_THRESHOLDS = {
    "low": 0.10,
    "medium": 0.25,
}


def classify_psi(psi_value: float) -> DriftLevel:
    if psi_value < PSI_THRESHOLDS["low"]:
        return "none"
    if psi_value < PSI_THRESHOLDS["medium"]:
        return "low"
    return "high"


# ============================================================
# Prediction drift
# ============================================================

PREDICTION_THRESHOLDS = {
    "mean_shift_std": 2.0,     # 2σ shift
    "p95_shift_ratio": 0.15,   # 15% tail movement
}


def classify_prediction_drift(
    mean_shift: float,
    baseline_std: float,
    p95_shift: float,
    baseline_p95: float,
) -> DriftLevel:

    mean_std_units = abs(mean_shift) / max(baseline_std, 1e-6)
    p95_ratio = abs(p95_shift) / max(baseline_p95, 1e-6)

    if (
        mean_std_units < PREDICTION_THRESHOLDS["mean_shift_std"]
        and p95_ratio < PREDICTION_THRESHOLDS["p95_shift_ratio"]
    ):
        return "none"

    if (
        mean_std_units < 3.0
        and p95_ratio < 0.25
    ):
        return "low"

    return "high"


# ============================================================
# Volume drift (sanity)
# ============================================================

VOLUME_THRESHOLDS = {
    "min_samples": 200,
    "drop_ratio": 0.5,
    "spike_ratio": 2.0,
}


def classify_volume_drift(
    baseline_n: int,
    current_n: int,
) -> DriftLevel:

    if current_n < VOLUME_THRESHOLDS["min_samples"]:
        return "high"  # insufficient data

    ratio = current_n / max(baseline_n, 1)

    if ratio < VOLUME_THRESHOLDS["drop_ratio"]:
        return "high"

    if ratio > VOLUME_THRESHOLDS["spike_ratio"]:
        return "medium"

    return "none"