from typing import Dict, Iterable
import pandas as pd


# -----------------------------
# Domain constraints
# -----------------------------

RANGE_CONSTRAINTS: Dict[str, tuple] = {
    "LIMIT_BAL": (0, None),       # credit limit >= 0
    "AGE": (18, 150),             # adult age range
}

CATEGORICAL_CODE_SETS: Dict[str, Iterable[int]] = {
    "SEX": {1, 2},
    "EDUCATION": {0, 1, 2, 3, 4, 5, 6},
    "MARRIAGE": {0, 1, 2, 3},
}

ORDINAL_MIN_MAX: Dict[str, tuple] = {
    "PAY_0": (-2, 9),
    "PAY_2": (-2, 9),
    "PAY_3": (-2, 9),
    "PAY_4": (-2, 9),
    "PAY_5": (-2, 9),
    "PAY_6": (-2, 9),
}


# -----------------------------
# Validation entry point
# -----------------------------

def validate_data(df: pd.DataFrame) -> None:

    errors = []

    # -------------------------
    # Missing value check
    # -------------------------
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]

    if not missing_cols.empty:
        errors.append(
            "Missing values detected:\n"
            + missing_cols.to_string()
        )

    # -------------------------
    # Numeric range checks
    # -------------------------
    for col, (min_val, max_val) in RANGE_CONSTRAINTS.items():
        if col not in df.columns:
            continue

        if min_val is not None:
            invalid = df[df[col] < min_val]
            if not invalid.empty:
                errors.append(
                    f"{col}: values below {min_val} detected"
                )

        if max_val is not None:
            invalid = df[df[col] > max_val]
            if not invalid.empty:
                errors.append(
                    f"{col}: values above {max_val} detected"
                )

    # -------------------------
    # Categorical code checks
    # -------------------------
    for col, valid_codes in CATEGORICAL_CODE_SETS.items():
        if col not in df.columns:
            continue

        invalid_codes = set(df[col].unique()) - valid_codes
        if invalid_codes:
            errors.append(
                f"{col}: invalid codes detected {invalid_codes}"
            )

    # -------------------------
    # Ordinal range checks
    # -------------------------
    for col, (min_val, max_val) in ORDINAL_MIN_MAX.items():
        if col not in df.columns:
            continue

        invalid = df[(df[col] < min_val) | (df[col] > max_val)]
        if not invalid.empty:
            errors.append(
                f"{col}: values outside [{min_val}, {max_val}] detected"
            )

    # -------------------------
    # Final decision
    # -------------------------
    if errors:
        raise ValueError(
            "Data validation failed:\n" + "\n".join(errors)
        )