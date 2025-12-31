from pathlib import Path
import argparse
import sys

import numpy as np

from src.models.tree_models import train_xgboost, train_lightgbm
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

ARTIFACTS_BASE_DIR = Path("artifacts/models")


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tree-based model for Phase 4"
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm"],
        required=True
    )
    parser.add_argument(
        "--num_leaves",
        type=int,
        default=31
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--min_child_samples",
        type=int,
        default=20
    )
    return parser.parse_args()


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    args = parse_args()

    try:
        print("Loading feature matrices...")
        X_train = np.load(FEATURES_DIR / "X_train.npy")
        X_val = np.load(FEATURES_DIR / "X_val.npy")

        print("Loading labels...")
        y_train = np.load(LABELS_DIR / "y_train.npy", allow_pickle=True).astype(int)
        y_val = np.load(LABELS_DIR / "y_val.npy", allow_pickle=True).astype(int)

        print(f"Training {args.model} model...")

        if args.model == "xgboost":
            model = train_xgboost(
                X_train,
                y_train,
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
            )
        elif args.model == "lightgbm":
            model = train_lightgbm(
                X_train,
                y_train,
                n_estimators=300,
                num_leaves=args.num_leaves,
                max_depth=args.max_depth,
                min_child_samples=args.min_child_samples,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
            )
        else:
            raise ValueError(f"Unsupported model type: {args.model}")

        print("Evaluating on validation set...")
        metrics = evaluate_binary_classifier(model, X_val, y_val)

        mean_pred, frac_pos = compute_calibration_data(
            model, X_val, y_val
        )

        model_dir = ARTIFACTS_BASE_DIR / args.model
        model_dir.mkdir(parents=True, exist_ok=True)

        print("Persisting model artifacts...")
        save_model(model, model_dir / "model.joblib")
        save_metrics(metrics, model_dir / "metrics.json")

        np.savez(
            model_dir / "calibration.npz",
            mean_predicted_value=mean_pred,
            fraction_of_positives=frac_pos,
        )

        print(f"\n{args.model.upper()} training completed.")
        print("Validation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    except Exception as e:
        print("\nTree model training failed.")
        print(f"Error: {e}")
        sys.exit(1)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()