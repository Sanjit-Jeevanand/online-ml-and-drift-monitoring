from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features.contracts import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
)


# ============================================================
# Ordinal feature ordering (explicit, enforced)
# ============================================================

ORDINAL_CATEGORIES: List[List[int]] = [
    list(range(-2, 10)),  # PAY_0
    list(range(-2, 10)),  # PAY_2
    list(range(-2, 10)),  # PAY_3
    list(range(-2, 10)),  # PAY_4
    list(range(-2, 10)),  # PAY_5
    list(range(-2, 10)),  # PAY_6
]


# ============================================================
# Preprocessing pipeline builder
# ============================================================

def build_preprocessing_pipeline() -> ColumnTransformer:

    continuous_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", continuous_pipeline, CONTINUOUS_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("ord", ordinal_pipeline, ORDINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor