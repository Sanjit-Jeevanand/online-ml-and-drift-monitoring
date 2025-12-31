from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from scipy.io import arff

from src.data.schema import validate_schema


# -----------------------------
# Column name mappings
# -----------------------------

RAW_TO_SEMANTIC_COLUMN_MAP: Dict[str, str] = {
    "x1": "LIMIT_BAL",
    "x2": "SEX",
    "x3": "EDUCATION",
    "x4": "MARRIAGE",
    "x5": "AGE",
    "x6": "PAY_0",
    "x7": "PAY_2",
    "x8": "PAY_3",
    "x9": "PAY_4",
    "x10": "PAY_5",
    "x11": "PAY_6",
    "x12": "BILL_AMT1",
    "x13": "BILL_AMT2",
    "x14": "BILL_AMT3",
    "x15": "BILL_AMT4",
    "x16": "BILL_AMT5",
    "x17": "BILL_AMT6",
    "x18": "PAY_AMT1",
    "x19": "PAY_AMT2",
    "x20": "PAY_AMT3",
    "x21": "PAY_AMT4",
    "x22": "PAY_AMT5",
    "x23": "PAY_AMT6",
}


# -----------------------------
# Loader
# -----------------------------


def load_openml_credit_default(
    arff_path: Path,
    apply_semantic_mapping: bool = True,
) -> pd.DataFrame:

    if not arff_path.exists():
        raise FileNotFoundError(f"ARFF file not found: {arff_path}")

    # Load ARFF
    raw_data, _ = arff.loadarff(arff_path)
    df = pd.DataFrame(raw_data)

    # Decode byte strings
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.decode("utf-8")

    # 🔒 VALIDATE RAW SCHEMA HERE (before renaming)
    validate_schema(df)

    # Apply semantic feature mapping
    if apply_semantic_mapping:
        df = df.rename(columns=RAW_TO_SEMANTIC_COLUMN_MAP)

    return df