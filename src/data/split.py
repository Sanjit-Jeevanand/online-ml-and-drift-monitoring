from pathlib import Path
from typing import Tuple

import pandas as pd


# -----------------------------
# Split configuration
# -----------------------------

TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

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

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by proxy time
    df_sorted = df.sort_values(by=time_column).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * TRAIN_FRACTION)
    val_end = train_end + int(n * VALIDATION_FRACTION)

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    # Persist immutable snapshots
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    return train_df, val_df, test_df