from pathlib import Path
import sys
import json

import joblib
import pandas as pd
import numpy as np

from src.features.preprocess import build_preprocessing_pipeline
from src.features.introspection import build_feature_metadata


# ============================================================
# Paths
# ============================================================

SPLITS_DIR = Path("data/interim/splits")

ARTIFACTS_DIR = Path("artifacts/features")

PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
METADATA_PATH = ARTIFACTS_DIR / "feature_metadata.json"

X_TRAIN_PATH = ARTIFACTS_DIR / "X_train.npy"
X_VAL_PATH = ARTIFACTS_DIR / "X_val.npy"
X_TEST_PATH = ARTIFACTS_DIR / "X_test.npy"


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    try:
        print("Loading data splits...")
        train_df = pd.read_csv(SPLITS_DIR / "train.csv")
        val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
        test_df = pd.read_csv(SPLITS_DIR / "test.csv")

        print("Building preprocessing pipeline...")
        preprocessor = build_preprocessing_pipeline()

        print("Fitting preprocessing on training data...")
        X_train = preprocessor.fit_transform(train_df)

        print("Transforming validation and test data...")
        X_val = preprocessor.transform(val_df)
        X_test = preprocessor.transform(test_df)

        print("Extracting feature metadata...")
        metadata = build_feature_metadata(preprocessor)

        print("Persisting feature artifacts...")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(preprocessor, PREPROCESSOR_PATH)

        np.save(X_TRAIN_PATH, X_train)
        np.save(X_VAL_PATH, X_val)
        np.save(X_TEST_PATH, X_test)

        with open(METADATA_PATH, "w") as f:
            json.dump(
                {
                    "version": metadata.version,
                    "n_features": metadata.n_features,
                    "feature_names": metadata.feature_names,
                    "feature_types": metadata.feature_types,
                },
                f,
                indent=2,
            )

        print("\nFeature build completed successfully.")
        print(f"Feature version: {metadata.version}")
        print(f"Number of features: {metadata.n_features}")
        print(f"Artifacts written to: {ARTIFACTS_DIR.resolve()}")

    except Exception as e:
        print("\nFeature build failed.")
        print(f"Error: {e}")
        sys.exit(1)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()