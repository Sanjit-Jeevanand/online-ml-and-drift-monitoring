from pathlib import Path
import json
from typing import Dict


# ============================================================
# Allowed retraining decisions
# ============================================================

RETRAIN_DECISIONS = {
    "RETRAIN_RECOMMENDED",
    "RETRAIN_REQUIRED",
}


# ============================================================
# Core trigger logic
# ============================================================

def load_monitoring_decision(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Monitoring decision file not found: {path}"
        )

    with path.open("r") as f:
        decision = json.load(f)

    if "decision" not in decision:
        raise ValueError(
            "Invalid decision file: missing 'decision' field"
        )

    return decision


def should_retrain(decision_payload: Dict) -> bool:
    decision = decision_payload.get("decision")

    return decision in RETRAIN_DECISIONS


def retraining_reason(decision_payload: Dict) -> str:
    reasons = decision_payload.get("reasons", [])

    if not reasons:
        return "Retraining triggered without explicit reasons."

    return "; ".join(reasons)