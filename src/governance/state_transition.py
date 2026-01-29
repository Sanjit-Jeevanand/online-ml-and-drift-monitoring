from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
import json


# ============================================================
# Paths
# ============================================================

GOVERNANCE_DIR = Path("artifacts/governance")

STATE_PATH = GOVERNANCE_DIR / "state.json"
DECISION_PATH = GOVERNANCE_DIR / "decision.json"
HISTORY_PATH = GOVERNANCE_DIR / "state_history.jsonl"


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _append_history(entry: Dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================
# Transition executor
# ============================================================

def apply_state_transition(**_ignored_kwargs) -> Dict[str, Any]:
    state = _load_json(STATE_PATH)
    decision = _load_json(DECISION_PATH)

    current_state = state.get("state")
    next_state = decision.get("next_state")
    decision_type = decision.get("decision")

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not current_state:
        raise ValueError("Current governance state missing.")

    if not next_state:
        raise ValueError("Decision missing next_state.")

    if current_state == next_state:
        return {
            "applied": False,
            "reason": "No-op transition (state unchanged)",
            "current_state": current_state,
        }

    if decision_type not in {"ADVANCE", "MANUAL_OVERRIDE"}:
        return {
            "applied": False,
            "reason": f"Decision '{decision_type}' does not permit transition",
            "current_state": current_state,
        }

    # --------------------------------------------------------
    # Apply transition
    # --------------------------------------------------------

    new_state = {
        "state": next_state,
        "model_name": state.get("model_name"),
        "model_version": state.get("model_version"),
        "updated_at": utc_now(),
        "updated_by": "governance_controller",
    }

    _write_json(STATE_PATH, new_state)

    # --------------------------------------------------------
    # Persist audit history
    # --------------------------------------------------------

    history_entry = {
        "from_state": current_state,
        "to_state": next_state,
        "decision": decision_type,
        "reasons": decision.get("reasons", []),
        "model_name": state.get("model_name"),
        "model_version": state.get("model_version"),
        "applied_at": utc_now(),
    }

    _append_history(history_entry)

    return {
        "applied": True,
        "from_state": current_state,
        "to_state": next_state,
        "model_name": state.get("model_name"),
        "model_version": state.get("model_version"),
    }


# ============================================================
# CLI entry point (optional but useful)
# ============================================================

if __name__ == "__main__":
    result = apply_state_transition()
    print("[Governance] State transition applied")
    print(json.dumps(result, indent=2))