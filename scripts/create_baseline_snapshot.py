from pathlib import Path
import json
import numpy as np
from datetime import datetime, timezone

from src.features.metadata import load_feature_metadata_file


# ============================================================
# Paths
# ============================================================

FEATURES_DIR = Path("artifacts/features")
METADATA_PATH = FEATURES_DIR / "feature_metadata.json"

SNAPSHOT_DIR = Path("snapshots")
BASELINE_SNAPSHOT_PATH = SNAPSHOT_DIR / "baseline.json"

REFERENCE_FEATURES_PATH = FEATURES_DIR / "X_val.npy"


# ============================================================
# Helpers
# ============================================================

def compute_feature_distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# ============================================================
# Main logic
# ============================================================

def main() -> None:
    print("Creating baseline snapshot...")

    if not REFERENCE_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Reference features not found: {REFERENCE_FEATURES_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature metadata not found: {METADATA_PATH}"
        )

    # Ensure output directory exists
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X = np.load(REFERENCE_FEATURES_PATH)
    metadata = load_feature_metadata_file(METADATA_PATH)

    feature_names = metadata.feature_names
    feature_types = metadata.feature_types

    if X.shape[1] != len(feature_names):
        raise ValueError(
            "Feature count mismatch between metadata and reference data"
        )

    # --------------------------------------------------------
    # Compute distributions
    # --------------------------------------------------------

    distributions = {}

    for idx, feature_name in enumerate(feature_names):
        # All model input features are numeric post-preprocessing
        values = X[:, idx]
        distributions[feature_name] = compute_feature_distribution(values)

    # --------------------------------------------------------
    # Build snapshot
    # --------------------------------------------------------

    snapshot = {
        "snapshot_type": "baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "lightgbm",
        "model_version": "v1.1.0",
        "n_samples": int(X.shape[0]),
        "feature_distributions": distributions,
    }

    # --------------------------------------------------------
    # Persist
    # --------------------------------------------------------

    with BASELINE_SNAPSHOT_PATH.open("w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Baseline snapshot written to: {BASELINE_SNAPSHOT_PATH}")
    print(f"Features baselined: {len(distributions)}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()