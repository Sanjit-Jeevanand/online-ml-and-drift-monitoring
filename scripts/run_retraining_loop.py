from pathlib import Path
import json
from datetime import datetime

from src.retraining.triggers import (
    load_monitoring_decision,
    should_retrain,
    retraining_reason,
)
from src.features.preprocess import load_preprocessor
from src.features.metadata import load_feature_metadata_file
from src.models.tree_models import train_lightgbm
from src.models.artifacts import save_model_artifacts

import pandas as pd
import numpy as np


DECISION_PATH = Path("artifacts/drift/decision.json")
DATA_PATH = Path("data/interim/validated/openml_credit_default.csv")
FEATURE_DIR = Path("artifacts/features")
OUTPUT_DIR = Path("artifacts/models/lightgbm/candidate")


def main() -> None:
    # --------------------------------------------------
    # 1. Check retraining trigger
    # --------------------------------------------------

    decision = load_monitoring_decision(DECISION_PATH)

    if not should_retrain(decision):
        print("No retraining triggered. Exiting.")
        return

    print("Retraining triggered:")
    print(retraining_reason(decision))

    # --------------------------------------------------
    # 2. Load data
    # --------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["default_flag", "id"])
    y = df["default_flag"]

    # --------------------------------------------------
    # 3. Load frozen preprocessing
    # --------------------------------------------------

    preprocessor = load_preprocessor(
        FEATURE_DIR / "preprocessor.joblib"
    )

    X_processed = preprocessor.transform(X)

    # --------------------------------------------------
    # 4. Train candidate model
    # --------------------------------------------------

    model = train_lightgbm(
        X_train=X_processed,
        y_train=y,
        params={
            "num_leaves": 31,
            "min_child_samples": 50,
            "random_state": 42,
        },
    )

    # --------------------------------------------------
    # 5. Persist candidate artifacts
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_model_artifacts(
        model=model,
        output_dir=OUTPUT_DIR,
        metadata={
            "trained_at": datetime.utcnow().isoformat(),
            "trigger_reason": retraining_reason(decision),
            "model_family": "lightgbm",
            "candidate": True,
        },
    )

    print("Candidate model trained and saved.")


if __name__ == "__main__":
    main()