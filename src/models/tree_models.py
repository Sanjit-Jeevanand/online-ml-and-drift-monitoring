from typing import Optional

import numpy as np

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# XGBoost
# ============================================================

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    min_child_weight: int = 1,
    random_state: int = 42,
    n_jobs: int = -1,
) -> XGBClassifier:

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# LightGBM
# ============================================================

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 300,
    max_depth: int = -1,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 0.0,
    min_child_samples: int = 20,
    random_state: int = 42,
    n_jobs: int = -1,
) -> LGBMClassifier:

    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_samples=min_child_samples,
        objective="binary",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(X_train, y_train)

    return model