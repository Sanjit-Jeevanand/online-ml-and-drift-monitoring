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

    baseline_numeric = baseline["features"]["numeric"]
    current_numeric = current["features"]["numeric"]

    for feature, base_stats in baseline_numeric.items():
        if feature not in current_numeric:
            continue

        # Reconstruct distributions approximately using quantiles
        # NOTE: snapshot-based approximation, not raw samples
        base_dist = _approximate_distribution(base_stats)
        curr_dist = _approximate_distribution(current_numeric[feature])

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

    drift_scores = {}

    baseline_cat = baseline["features"]["categorical"]
    current_cat = current["features"]["categorical"]

    for feature, base_data in baseline_cat.items():
        if feature not in current_cat:
            continue

        base_freqs = base_data["frequencies"]
        curr_freqs = current_cat[feature]["frequencies"]

        psi = 0.0

        all_keys = set(base_freqs.keys()) | set(curr_freqs.keys())

        for k in all_keys:
            b = base_freqs.get(k, epsilon)
            c = curr_freqs.get(k, epsilon)

            psi += (c - b) * np.log(c / b)

        drift_scores[feature] = float(psi)

    return drift_scores


# ============================================================
# Prediction drift
# ============================================================

def prediction_drift(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
) -> Dict[str, float]:

    base_pred = baseline.get("predictions")
    curr_pred = current.get("predictions")

    if not base_pred or not curr_pred:
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

    base_n = baseline["volume"]["n_requests"]
    curr_n = current["volume"]["n_requests"]

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