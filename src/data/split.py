from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np


# -----------------------------
# Split configuration
# -----------------------------

TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

LABELS_OUTPUT_DIR = Path("artifacts/labels")

assert abs(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION - 1.0) < 1e-6


# -----------------------------
# Temporal split function
# -----------------------------

def temporal_split(
    df: pd.DataFrame,
    output_dir: Path,
    time_column: str = "id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if time_column not in df.columns:
        raise ValueError(
            f"Time column '{time_column}' not found in DataFrame"
        )
    if "y" not in df.columns:
        raise ValueError(
            "Target column 'y' not found in DataFrame"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by proxy time
    df_sorted = df.sort_values(by=time_column).reset_index(drop=True)

    y = df_sorted["y"].values
    X = df_sorted.drop(columns=["y"])

    n = len(df_sorted)
    train_end = int(n * TRAIN_FRACTION)
    val_end = train_end + int(n * VALIDATION_FRACTION)

    train_df = X.iloc[:train_end].copy()
    val_df = X.iloc[train_end:val_end].copy()
    test_df = X.iloc[val_end:].copy()

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    # Persist immutable snapshots
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    LABELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(LABELS_OUTPUT_DIR / "y_train.npy", y_train)
    np.save(LABELS_OUTPUT_DIR / "y_val.npy", y_val)
    np.save(LABELS_OUTPUT_DIR / "y_test.npy", y_test)

    return train_df, val_df, test_df