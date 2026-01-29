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


# ============================================================
# High-level artifact helpers
# ============================================================

def save_model_artifacts(
    model: Any,
    metrics: Dict[str, float],
    base_dir: Path,
    *,
    calibration: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, base_dir / "model.joblib")
    save_metrics(metrics, base_dir / "metrics.json")

    if calibration is not None:
        serializable_calibration = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in calibration.items()
        }

        with open(base_dir / "calibration.json", "w") as f:
            json.dump(serializable_calibration, f, indent=2)

    if metadata is not None:
        with open(base_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def load_model_artifacts(
    base_dir: Path,
) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {
        "model": load_model(base_dir / "model.joblib"),
        "metrics": load_metrics(base_dir / "metrics.json"),
    }

    calibration_path = base_dir / "calibration.json"
    if calibration_path.exists():
        with open(calibration_path, "r") as f:
            artifacts["calibration"] = json.load(f)

    metadata_path = base_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            artifacts["metadata"] = json.load(f)

    return artifacts