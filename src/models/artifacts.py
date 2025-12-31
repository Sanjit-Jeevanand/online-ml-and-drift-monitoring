from pathlib import Path
from typing import Any, Dict

import json
import joblib


# ============================================================
# Model persistence
# ============================================================

def save_model(
    model: Any,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(
    path: Path,
) -> Any:

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    return joblib.load(path)


# ============================================================
# Metrics persistence
# ============================================================

def save_metrics(
    metrics: Dict[str, float],
    path: Path,
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(
    path: Path,
) -> Dict[str, float]:
    
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)