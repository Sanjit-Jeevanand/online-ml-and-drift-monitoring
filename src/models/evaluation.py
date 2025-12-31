from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


def evaluate_binary_classifier(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
) -> Dict[str, float]:

    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "brier_score": brier_score_loss(y, y_proba),
    }

    return metrics


def compute_calibration_data(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    
    y_proba = model.predict_proba(X)[:, 1]

    frac_pos, mean_pred = calibration_curve(
        y,
        y_proba,
        n_bins=n_bins,
        strategy="uniform",
    )

    return mean_pred, frac_pos