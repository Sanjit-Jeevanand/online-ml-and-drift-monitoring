"""
End-to-end data ingestion pipeline for the OpenML Credit Default dataset.

This script is the ONLY sanctioned entry point for data ingestion.
It performs:
- raw data loading
- schema validation
- data integrity validation
- temporal splitting
- immutable snapshot persistence

If this script fails, downstream phases must not proceed.
"""

from pathlib import Path
import sys

from src.data.load import load_openml_credit_default
from src.data.validate import validate_data
from src.data.split import temporal_split


# -----------------------------
# Paths
# -----------------------------

RAW_DATA_PATH = Path(
    "data/raw/openml_credit_default/credit_default.arff"
)

VALIDATED_OUTPUT_PATH = Path(
    "data/interim/validated/openml_credit_default.csv"
)

SPLITS_OUTPUT_DIR = Path(
    "data/interim/splits"
)


# -----------------------------
# Main pipeline
# -----------------------------

def main() -> None:
    try:
        print("Loading raw dataset...")
        df = load_openml_credit_default(RAW_DATA_PATH)


        print("Validating data integrity...")
        validate_data(df)

        print("Saving validated dataset...")
        VALIDATED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(VALIDATED_OUTPUT_PATH, index=False)

        print("Performing temporal split...")
        temporal_split(
            df=df,
            output_dir=SPLITS_OUTPUT_DIR,
            time_column="id",
        )

        print("\nData ingestion completed successfully.")
        print(f"Validated data saved to: {VALIDATED_OUTPUT_PATH}")
        print(f"Splits saved to: {SPLITS_OUTPUT_DIR.resolve()}")

    except Exception as e:
        print("\nData ingestion failed.")
        print(f"Error: {e}")
        sys.exit(1)


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    main()