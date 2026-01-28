from typing import Dict, Any, List, Literal

from src.monitoring.thresholds import (
    classify_ks,
    classify_psi,
    classify_prediction_drift,
    classify_volume_drift,
)

# ============================================================
# Decision types
# ============================================================

Decision = Literal[
    "NO_ACTION",
    "MONITOR",
    "RETRAIN_RECOMMENDED",
    "RETRAIN_REQUIRED",
]


# ============================================================
# Core decision function
# ============================================================

def decide_drift_action(
    *,
    ks_metrics: Dict[str, float],
    psi_metrics: Dict[str, float],
    prediction_metrics: Dict[str, float],
    baseline_prediction_stats: Dict[str, Any],
    baseline_volume: int,
    current_volume: int,
) -> Dict[str, Any]:

    reasons: List[str] = []

    # --------------------------------------------------------
    # 1. Volume sanity check
    # --------------------------------------------------------

    volume_level = classify_volume_drift(
        baseline_n=baseline_volume,
        current_n=current_volume,
    )

    if volume_level == "high":
        return _decision(
            "MONITOR",
            reasons=["Insufficient or unstable traffic volume"],
            severity="high",
        )

    # --------------------------------------------------------
    # 2. Prediction drift (highest priority)
    # --------------------------------------------------------

    if prediction_metrics:
        pred_level = classify_prediction_drift(
            mean_shift=prediction_metrics["mean_shift"],
            baseline_std=baseline_prediction_stats["std"],
            p95_shift=prediction_metrics["p95_shift"],
            baseline_p95=baseline_prediction_stats["quantiles"]["p95"],
        )

        if pred_level == "high":
            return _decision(
                "RETRAIN_REQUIRED",
                reasons=["Severe prediction distribution shift"],
                severity="high",
            )

        if pred_level == "low":
            reasons.append("Moderate prediction distribution shift")

    # --------------------------------------------------------
    # 3. Feature drift aggregation
    # --------------------------------------------------------

    high_drift_features = []
    moderate_drift_features = []

    for feature, ks in ks_metrics.items():
        level = classify_ks(ks)
        if level == "high":
            high_drift_features.append(feature)
        elif level == "low":
            moderate_drift_features.append(feature)

    for feature, psi in psi_metrics.items():
        level = classify_psi(psi)
        if level == "high":
            high_drift_features.append(feature)
        elif level == "low":
            moderate_drift_features.append(feature)

    if len(high_drift_features) >= 2:
        return _decision(
            "RETRAIN_REQUIRED",
            reasons=[
                f"Severe drift in multiple features: {high_drift_features}"
            ],
            severity="high",
        )

    if high_drift_features:
        return _decision(
            "RETRAIN_RECOMMENDED",
            reasons=[
                f"Severe drift detected in feature(s): {high_drift_features}"
            ],
            severity="medium",
        )

    if len(moderate_drift_features) >= 3:
        return _decision(
            "MONITOR",
            reasons=[
                f"Moderate drift accumulating in features: {moderate_drift_features}"
            ],
            severity="low",
        )

    # --------------------------------------------------------
    # 4. Default
    # --------------------------------------------------------

    return _decision(
        "NO_ACTION",
        reasons=["No significant drift detected"],
        severity="none",
    )


# ============================================================
# Helpers
# ============================================================

def _decision(
    decision: Decision,
    *,
    reasons: List[str],
    severity: str,
) -> Dict[str, Any]:
    return {
        "decision": decision,
        "severity": severity,
        "reasons": reasons,
    }