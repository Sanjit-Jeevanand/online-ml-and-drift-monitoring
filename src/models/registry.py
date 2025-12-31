from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json
import joblib


# ============================================================
# Registry base path
# ============================================================

REGISTRY_BASE_DIR = Path("artifacts/models")


# ============================================================
# Helpers
# ============================================================

def _version_dir(model_name: str, version: str) -> Path:
    """
    Resolve the directory for a given model name and version.
    """
    return REGISTRY_BASE_DIR / model_name / version


def _ensure_not_exists(path: Path) -> None:
    """
    Prevent accidental overwrites of immutable artifacts.
    """
    if path.exists():
        raise FileExistsError(
            f"Model version already exists and is immutable: {path}"
        )


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ============================================================
# Registration
# ============================================================

def register_model(
    *,
    model_name: str,
    version: str,
    model,
    preprocessor,
    metrics: Dict[str, float],
    calibration: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Path:
    """
    Register a trained model and all associated artifacts
    under a semantic version.

    This operation is IMMUTABLE.
    """

    version_path = _version_dir(model_name, version)
    _ensure_not_exists(version_path)

    version_path.mkdir(parents=True, exist_ok=False)

    # --------------------------------------------------------
    # Save core artifacts
    # --------------------------------------------------------

    joblib.dump(model, version_path / "model.joblib")
    joblib.dump(preprocessor, version_path / "preprocessor.joblib")

    with open(version_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # calibration expected as numpy-friendly dict
    # e.g. {"mean_predicted_value": ..., "fraction_of_positives": ...}
    import numpy as np
    np.savez(version_path / "calibration.npz", **calibration)

    # --------------------------------------------------------
    # Metadata enrichment
    # --------------------------------------------------------

    enriched_metadata = {
        **metadata,
        "model_name": model_name,
        "model_version": version,
        "registered_at": _utc_now(),
    }

    with open(version_path / "metadata.json", "w") as f:
        json.dump(enriched_metadata, f, indent=2)

    # Optional human-readable summary
    with open(version_path / "README.md", "w") as f:
        f.write(f"# Model Artifact\n\n")
        f.write(f"- Model: {model_name}\n")
        f.write(f"- Version: {version}\n")
        f.write(f"- Registered at: {enriched_metadata['registered_at']}\n")

    return version_path


# ============================================================
# Loading
# ============================================================

def load_model(
    *,
    model_name: str,
    version: str,
):
    """
    Load a specific model version and its preprocessor.
    """

    version_path = _version_dir(model_name, version)

    if not version_path.exists():
        raise FileNotFoundError(
            f"Requested model version does not exist: {version_path}"
        )

    model = joblib.load(version_path / "model.joblib")
    preprocessor = joblib.load(version_path / "preprocessor.joblib")

    return model, preprocessor


def load_metadata(
    *,
    model_name: str,
    version: str,
) -> Dict[str, Any]:
    """
    Load metadata for a specific model version.
    """

    version_path = _version_dir(model_name, version)
    metadata_path = version_path / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata for model version: {version_path}"
        )

    with open(metadata_path, "r") as f:
        return json.load(f)


def list_versions(model_name: str) -> list[str]:
    """
    List all registered versions for a given model.
    """

    model_dir = REGISTRY_BASE_DIR / model_name
    if not model_dir.exists():
        return []

    return sorted(
        p.name for p in model_dir.iterdir() if p.is_dir()
    )