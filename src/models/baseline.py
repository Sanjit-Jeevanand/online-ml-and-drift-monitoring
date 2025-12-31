from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
) -> LogisticRegression:

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    return model