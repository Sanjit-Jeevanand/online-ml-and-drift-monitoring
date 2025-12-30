from typing import Dict

from src.features.contracts import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
)
from src.features.metadata import FeatureMetadata
from src.features.versioning import compute_feature_version


def build_feature_metadata(
    preprocessor,
) -> FeatureMetadata:

    feature_names = preprocessor.get_feature_names_out().tolist()

    feature_types: Dict[str, str] = {}

    for name in feature_names:
        if name.startswith(tuple(CONTINUOUS_FEATURES)):
            feature_types[name] = "continuous"
        elif name.startswith(tuple(CATEGORICAL_FEATURES)):
            feature_types[name] = "categorical"
        elif name.startswith(tuple(ORDINAL_FEATURES)):
            feature_types[name] = "ordinal"
        else:
            feature_types[name] = "unknown"

    contract = {
        "continuous": CONTINUOUS_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "ordinal": ORDINAL_FEATURES,
    }

    version = compute_feature_version(contract)

    return FeatureMetadata(
        version=version,
        feature_names=feature_names,
        feature_types=feature_types,
        n_features=len(feature_names),
    )