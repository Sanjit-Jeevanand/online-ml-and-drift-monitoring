from pathlib import Path
import json
import sys

import pandas as pd

from src.inference.predictor import Predictor


# ============================================================
# Paths
# ============================================================

FEATURES_DIR = Path("artifacts/features")
LABELS_DIR = Path("artifacts/labels")
DATA_SPLITS_DIR = Path("data/interim/splits")
CONFIG_DIR = Path("config")


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    try:
        # ----------------------------------------------------
        # Load production model pointer
        # ----------------------------------------------------

        prod_config_path = CONFIG_DIR / "production_model.json"

        if not prod_config_path.exists():
            raise FileNotFoundError(
                "Missing production model pointer: "
                "config/production_model.json"
            )

        with open(prod_config_path, "r") as f:
            prod_cfg = json.load(f)

        model_name = prod_cfg["model_name"]
        model_version = prod_cfg["model_version"]

        print("Loading predictor...")
        predictor = Predictor(
            model_name=model_name,
            model_version=model_version,
        )

        # ----------------------------------------------------
        # Load a real validation example
        # ----------------------------------------------------

        print("Loading validation data...")
        val_df = pd.read_csv(DATA_SPLITS_DIR / "validation.csv")

        example = val_df.iloc[0]

        # Drop target column if present
        if "default_flag" in example:
            example = example.drop("default_flag")

        raw_input = example.to_dict()

        print("\nRaw input example:")
        for k, v in list(raw_input.items())[:6]:
            print(f"  {k}: {v}")
        print("  ...")

        # ----------------------------------------------------
        # Run inference
        # ----------------------------------------------------

        proba = predictor.predict_proba(raw_input)

        print("\nInference result:")
        print(f"  Model: {model_name}:{model_version}")
        print(f"  Predicted probability: {proba:.6f}")

        print("\nLocal inference test PASSED.")

    except Exception as e:
        print("\nLocal inference test FAILED.")
        print(f"Error: {e}")
        sys.exit(1)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()