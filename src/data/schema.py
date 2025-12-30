from typing import List
import pandas as pd


# -----------------------------
# Core column roles
# -----------------------------

TARGET_COLUMN = "y"

IDENTIFIER_COLUMNS: List[str] = [
    "id",
]

# -----------------------------
# Feature definitions (raw)
# -----------------------------

# x1–x23 are all features in the OpenML ARFF
FEATURE_COLUMNS: List[str] = [
    f"x{i}" for i in range(1, 24)
]

# -----------------------------
# Forbidden columns
# -----------------------------

FORBIDDEN_COLUMNS: List[str] = [
    *IDENTIFIER_COLUMNS,
]

# -----------------------------
# Schema validation
# -----------------------------

def validate_schema(df: pd.DataFrame) -> None:

    errors = []

    # Target check
    if TARGET_COLUMN not in df.columns:
        errors.append(f"Missing target column: {TARGET_COLUMN}")

    # Feature presence
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing feature column: {col}")

    # Identifier leakage
    for col in IDENTIFIER_COLUMNS:
        if col in df.columns:
            errors.append(f"Identifier column present: {col}")

    if errors:
        raise ValueError(
            "Schema validation failed:\n" + "\n".join(errors)
        )