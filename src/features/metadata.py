from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class FeatureMetadata:

    version: str
    feature_names: List[str]
    feature_types: Dict[str, str]  
    n_features: int