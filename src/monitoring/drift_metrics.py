from typing import Dict, Any
import numpy as np
from scipy.stats import ks_2samp


# ============================================================
# Numeric drift metrics
# ============================================================

def ks_drift(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute Kolmogorov–Smirnov drift for numeric features.
    """

    drift_scores = {}

    # -------------------------------
    # Resolve baseline numeric stats
    # -------------------------------
    baseline_numeric = baseline.get("feature_distributions")

    if baseline_numeric is None:
        baseline_numeric = (
            baseline.get("features", {})
            .get("numeric")
        )

    if baseline_numeric is None:
        raise KeyError(
            "Baseline snapshot missing numeric feature distributions"
        )

    # -------------------------------
    # Resolve current numeric stats
    # -------------------------------
    current_numeric = current.get("feature_distributions")

    if current_numeric is None:
        current_numeric = (
            current.get("features", {})
            .get("numeric")
        )

    if current_numeric is None:
        raise KeyError(
            "Current snapshot missing numeric feature distributions"
        )

    # -------------------------------
    # Compute KS per feature
    # -------------------------------
    for feature, base_stats in baseline_numeric.items():
        curr_stats = current_numeric.get(feature)
        if curr_stats is None:
            continue

        base_dist = _approximate_distribution(base_stats)
        curr_dist = _approximate_distribution(curr_stats)

        ks_stat, _ = ks_2samp(base_dist, curr_dist)
        drift_scores[feature] = float(ks_stat)

    return drift_scores


# ============================================================
# Categorical drift metrics
# ============================================================

def psi_drift(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute Population Stability Index (PSI) for categorical features.
    """

    return {}


# ============================================================
# Prediction drift
# ============================================================

def prediction_drift(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
) -> Dict[str, float]:

    base_pred = baseline.get("predictions")
    curr_pred = current.get("predictions")

    # Prediction drift is optional; skip if unavailable
    if base_pred is None or curr_pred is None:
        return {}

    return {
        "mean_shift": curr_pred["mean"] - base_pred["mean"],
        "std_shift": curr_pred["std"] - base_pred["std"],
        "p95_shift": (
            curr_pred["quantiles"]["p95"]
            - base_pred["quantiles"]["p95"]
        ),
    }


# ============================================================
# Volume drift (sanity check)
# ============================================================

def volume_drift(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute volume drift based on request counts.
    If volume information is missing, skip volume drift safely.
    """

    base_volume = baseline.get("volume")
    curr_volume = current.get("volume")

    if not base_volume or not curr_volume:
        return {}

    base_n = base_volume.get("n_requests", 0)
    curr_n = curr_volume.get("n_requests", 0)

    if base_n == 0:
        return {"ratio": float("inf")}

    return {
        "ratio": curr_n / base_n
    }


# ============================================================
# Helpers
# ============================================================

def _approximate_distribution(stats: Dict[str, Any], n: int = 1000):

    mean = stats["mean"]
    std = stats["std"]

    if std == 0:
        return np.full(n, mean)

    return np.random.normal(mean, std, size=n)