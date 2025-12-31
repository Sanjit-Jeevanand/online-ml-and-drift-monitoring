from pathlib import Path
import sys

import numpy as np

from src.models.baseline import train_logistic_regression
from src.models.evaluation import (
    evaluate_binary_classifier,
    compute_calibration_data,
)
from src.models.artifacts import save_model, save_metrics


# ============================================================
# Paths
# ============================================================

FEATURES_DIR = Path("artifacts/features")
LABELS_DIR = Path("artifacts/labels")

ARTIFACTS_DIR = Path("artifacts/models/baseline")

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CALIBRATION_PATH = ARTIFACTS_DIR / "calibration.npz"


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    try:
        print("Loading feature matrices...")
        X_train = np.load(FEATURES_DIR / "X_train.npy")
        X_val = np.load(FEATURES_DIR / "X_val.npy")

        print("Loading labels...")
        y_train = np.load(LABELS_DIR / "y_train.npy", allow_pickle=True)
        y_val = np.load(LABELS_DIR / "y_val.npy", allow_pickle=True)
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)

        print("Training logistic regression baseline...")
        model = train_logistic_regression(X_train, y_train)

        print("Evaluating on validation set...")
        metrics = evaluate_binary_classifier(model, X_val, y_val)

        mean_pred, frac_pos = compute_calibration_data(
            model, X_val, y_val
        )

        print("Persisting model artifacts...")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        save_model(model, MODEL_PATH)
        save_metrics(metrics, METRICS_PATH)

        np.savez(
            CALIBRATION_PATH,
            mean_predicted_value=mean_pred,
            fraction_of_positives=frac_pos,
        )

        print("\nBaseline training completed.")
        print("Validation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    except Exception as e:
        print("\nBaseline training failed.")
        print(f"Error: {e}")
        sys.exit(1)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()