from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np

from src.models.registry import register_model


# ============================================================
# Paths
# ============================================================

ARTIFACTS_DIR = Path("artifacts")

FEATURES_DIR = ARTIFACTS_DIR / "features"
MODELS_DIR = ARTIFACTS_DIR / "models"


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a trained model into the model registry"
    )

    parser.add_argument(
        "--model-name",
        required=True,
    )

    parser.add_argument(
        "--version",
        required=True,
    )

    parser.add_argument(
        "--training-end-index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--baseline-version",
        default=None,
    )

    return parser.parse_args()


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    args = parse_args()

    try:
        model_dir = MODELS_DIR / args.model_name

        print("Loading trained model...")
        model = joblib.load(model_dir / "model.joblib")

        print("Loading preprocessing pipeline...")
        preprocessor = joblib.load(
            FEATURES_DIR / "preprocessor.joblib"
        )

        print("Loading evaluation metrics...")
        with open(model_dir / "metrics.json", "r") as f:
            metrics = json.load(f)

        print("Loading calibration data...")
        calib_npz = np.load(model_dir / "calibration.npz")
        calibration = {
            "mean_predicted_value": calib_npz["mean_predicted_value"],
            "fraction_of_positives": calib_npz["fraction_of_positives"],
        }

        print("Loading feature metadata...")
        with open(FEATURES_DIR / "feature_metadata.json", "r") as f:
            feature_metadata = json.load(f)

        # ----------------------------------------------------
        # Construct lineage metadata
        # ----------------------------------------------------

        metadata = {
            "training_data": {
                "source": "openml_credit_default",
                "end_index": args.training_end_index,
            },
            "feature_contract": {
                "version": feature_metadata["version"],
                "n_features": feature_metadata["n_features"],
            },
            "hyperparameters": getattr(model, "get_params", lambda: {})(),
        }

        if args.baseline_version is not None:
            metadata["baseline_comparison"] = {
                "baseline_version": args.baseline_version,
                "roc_auc_delta": None,  # optional: filled offline
                "pr_auc_delta": None,
            }

        print("Registering model artifact...")
        path = register_model(
            model_name=args.model_name,
            version=args.version,
            model=model,
            preprocessor=preprocessor,
            metrics=metrics,
            calibration=calibration,
            metadata=metadata,
        )

        print("\nModel successfully registered.")
        print(f"Location: {path}")

    except Exception as e:
        print("\nModel registration failed.")
        print(f"Error: {e}")
        sys.exit(1)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()