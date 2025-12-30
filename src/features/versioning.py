import hashlib
import json
from typing import Dict, List


def compute_feature_version(contract: Dict[str, List[str]]) -> str:
    serialized = json.dumps(contract, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]