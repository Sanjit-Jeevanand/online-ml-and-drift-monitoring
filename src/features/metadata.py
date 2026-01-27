from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class FeatureMetadata:

    version: str
    feature_names: List[str]
    feature_types: Dict[str, str]  
    n_features: int

import json
from pathlib import Path
from typing import Any


def load_feature_metadata_file(path: Path) -> FeatureMetadata:

    if not path.exists():
        raise FileNotFoundError(
            f"Feature metadata file not found: {path}"
        )

    with open(path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    required_keys = {"version", "feature_names", "feature_types", "n_features"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(
            f"Feature metadata missing required keys: {missing}"
        )

    return FeatureMetadata(
        version=data["version"],
        feature_names=data["feature_names"],
        feature_types=data["feature_types"],
        n_features=data["n_features"],
    )