from pathlib import Path
import json
import sys
from datetime import datetime, timezone

import numpy as np

from src.models.tree_models import train_lightgbm
from src.models.evaluation import (
    evaluate_binary_classifier,
    compute_calibration_data,
)
from src.models.artifacts import save_model_artifacts, load_model

# --------------------------------------------------
# Paths
# --------------------------------------------------

RETRAIN_SIGNAL_PATH = Path("artifacts/drift/retrain_signal.json")
FEATURES_DIR = Path("artifacts/features")
LABELS_DIR = Path("artifacts/labels")
CANDIDATE_DIR = Path("artifacts/models/candidate")
PROD_CONFIG_PATH = Path("config/production_model.json")

# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    print("[Candidate Training] Starting...")

    # --------------------------------------------------
    # 1. Gate check
    # --------------------------------------------------

    if not RETRAIN_SIGNAL_PATH.exists():
        print("[Candidate Training] No retraining signal found. Exiting.")
        sys.exit(0)

    retrain_signal = json.loads(RETRAIN_SIGNAL_PATH.read_text())

    print("[Candidate Training] Retraining authorized.")
    print(f"[Candidate Training] Reason(s): {retrain_signal.get('reasons')}")

    # --------------------------------------------------
    # 2a. Load production preprocessor
    # --------------------------------------------------

    if not PROD_CONFIG_PATH.exists():
        print("[Candidate Training] Missing production model config.")
        sys.exit(1)

    prod_cfg = json.loads(PROD_CONFIG_PATH.read_text())
    prod_model_name = prod_cfg["model_name"]
    prod_model_version = prod_cfg["model_version"]

    prod_model_dir = Path("artifacts/models") / prod_model_name / prod_model_version
    preprocessor_path = prod_model_dir / "preprocessor.joblib"

    if not preprocessor_path.exists():
        print("[Candidate Training] Production preprocessor not found.")
        sys.exit(1)

    # Load production metadata (for lineage validation only)
    prod_metadata_path = prod_model_dir / "metadata.json"
    if not prod_metadata_path.exists():
        raise RuntimeError("Production metadata is required for candidate training.")

    prod_metadata = json.loads(prod_metadata_path.read_text())
    prod_feature_contract = prod_metadata.get("feature_contract")
    if prod_feature_contract is None:
        raise RuntimeError("Production metadata missing 'feature_contract'; lineage is broken.")

    # Load feature contract from features/feature_metadata.json
    feature_metadata_path = FEATURES_DIR / "feature_metadata.json"
    if not feature_metadata_path.exists():
        raise RuntimeError("Feature metadata file missing; cannot determine feature contract.")
    feature_metadata = json.loads(feature_metadata_path.read_text())
    feature_contract = feature_metadata.get("version")
    if feature_contract is None:
        raise RuntimeError("Feature metadata missing 'version' (feature contract hash).")

    # Optional: warn if contract does not match prod lineage
    if feature_contract != prod_feature_contract:
        print("[Candidate Training] WARNING: Feature contract hash from feature_metadata.json does not match production metadata. Lineage may be broken.")

    preprocessor = load_model(preprocessor_path)

    # --------------------------------------------------
    # 2. Load training data
    # --------------------------------------------------

    try:
        X_train = np.load(FEATURES_DIR / "X_train.npy", allow_pickle=True)
        X_val = np.load(FEATURES_DIR / "X_val.npy", allow_pickle=True)
        y_train = np.load(LABELS_DIR / "y_train.npy", allow_pickle=True)
        y_val = np.load(LABELS_DIR / "y_val.npy", allow_pickle=True)
    except Exception as e:
        print(f"[Candidate Training] Failed to load training data: {e}")
        sys.exit(1)

    # Ensure labels are numeric (LightGBM + sklearn expect ints, not strings)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    if X_train.shape[0] != y_train.shape[0]:
        print("[Candidate Training] Feature/label row mismatch.")
        sys.exit(1)

    # --------------------------------------------------
    # 3. Train candidate model
    # --------------------------------------------------

    print("[Candidate Training] Training LightGBM candidate model...")

    model = train_lightgbm(
        X_train=X_train,
        y_train=y_train,
    )

    # --------------------------------------------------
    # 4. Evaluate candidate
    # --------------------------------------------------

    print("[Candidate Training] Evaluating candidate model...")

    metrics = evaluate_binary_classifier(
        model=model,
        X=X_val,
        y=y_val,
    )

    mean_pred, frac_pos = compute_calibration_data(
        model=model,
        X=X_val,
        y=y_val,
    )

    calibration = {
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
    }

    # --------------------------------------------------
    # 5. Persist candidate artifacts
    # --------------------------------------------------

    candidate_version = f"candidate-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    metadata = {
        "training_data": {
            "source": prod_metadata["training_data"]["source"],
            "end_index": prod_metadata["training_data"]["end_index"],
        },
        "feature_contract": {
            "version": feature_metadata["version"],
            "n_features": feature_metadata["n_features"],
        },
        "hyperparameters": model.get_params(),
        "model_name": "lightgbm",
        "candidate_version": candidate_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"[Candidate Training] Saving artifacts to {CANDIDATE_DIR}...")

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    save_model_artifacts(
        model=model,
        metrics=metrics,
        calibration=calibration,
        base_dir=CANDIDATE_DIR,
        metadata=metadata,
        preprocessor=preprocessor,
    )

    readme = (
        "Candidate Model Artifact\n"
        "========================\n\n"
        f"Version: {candidate_version}\n"
        f"Trained at: {metadata['trained_at']}\n"
        f"Reason: {retrain_signal.get('decision')}\n"
    )

    (CANDIDATE_DIR / "README.md").write_text(readme)

    print("[Candidate Training] Candidate model trained successfully.")


if __name__ == "__main__":
    main()
