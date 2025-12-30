"""
Feature contract for the credit default prediction system.

This file is the single source of truth for:
- which columns are features
- their semantic types
- which columns are forbidden
- what is allowed to reach the model

All training, inference, and monitoring code must respect this contract.
"""

from typing import List, Dict


# ============================================================
# Target
# ============================================================

TARGET_COLUMN: str = "default_flag"


# ============================================================
# Forbidden columns
# ============================================================

# Columns that must NEVER reach the model
FORBIDDEN_COLUMNS: List[str] = [
    "id",            # identifier
    TARGET_COLUMN,   # label
]


# ============================================================
# Semantic feature groups
# ============================================================

# ---------
# Continuous numeric features
# ---------
CONTINUOUS_FEATURES: List[str] = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# ---------
# Nominal categorical features (unordered)
# ---------
CATEGORICAL_FEATURES: List[str] = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
]

# ---------
# Ordinal features (ordered categories)
# ---------
ORDINAL_FEATURES: List[str] = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]


# ============================================================
# Derived authoritative lists
# ============================================================

ALL_FEATURES: List[str] = (
    CONTINUOUS_FEATURES
    + CATEGORICAL_FEATURES
    + ORDINAL_FEATURES
)

ALL_KNOWN_COLUMNS: List[str] = (
    ALL_FEATURES
    + FORBIDDEN_COLUMNS
)


# ============================================================
# Sanity checks (fail fast)
# ============================================================

def _assert_no_overlap() -> None:
    """Ensure no column appears in multiple semantic groups."""
    groups = {
        "continuous": set(CONTINUOUS_FEATURES),
        "categorical": set(CATEGORICAL_FEATURES),
        "ordinal": set(ORDINAL_FEATURES),
        "forbidden": set(FORBIDDEN_COLUMNS),
    }

    for name_a, set_a in groups.items():
        for name_b, set_b in groups.items():
            if name_a == name_b:
                continue
            overlap = set_a & set_b
            if overlap:
                raise ValueError(
                    f"Feature contract error: columns {overlap} "
                    f"appear in both '{name_a}' and '{name_b}'"
                )


_assert_no_overlap()