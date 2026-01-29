# scripts/execute_promotion.py

from pathlib import Path
import json
import sys
from datetime import datetime, timezone
import os

from src.governance.state_transition import apply_state_transition


GOVERNANCE_DIR = Path("artifacts/governance")
DECISION_PATH = GOVERNANCE_DIR / "decision.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    if not DECISION_PATH.exists():
        print("[Auto Executor] No decision artifact found. Exiting.")
        return

    decision = json.loads(DECISION_PATH.read_text())

    decision_type = decision.get("decision")
    current_state = decision.get("current_state")
    next_state = decision.get("next_state")
    model_name = decision.get("model_name")
    model_version = decision.get("model_version")

    print(f"[Auto Executor] Decision: {decision_type}")

    # --------------------------------------------------
    # Only act on explicit authority
    # --------------------------------------------------

    if decision_type not in {"AUTO_ADVANCE", "MANUAL_OVERRIDE"}:
        print("[Auto Executor] No executable transition. Exiting.")
        return

    # Allow MANUAL_OVERRIDE to execute even if state does not change
    if not next_state:
        print("[Auto Executor] Invalid transition (missing next_state). Exiting.")
        return

    if next_state == current_state and decision_type != "MANUAL_OVERRIDE":
        print("[Auto Executor] Invalid or no-op transition. Exiting.")
        return

    # --------------------------------------------------
    # Apply transition
    # --------------------------------------------------

    print(
        f"[Auto Executor] Applying transition: "
        f"{current_state} → {next_state}"
    )

    result = apply_state_transition()

    print("[Auto Executor] Result:")
    print(json.dumps(result, indent=2))

    # --------------------------------------------------
    # Execute promotion if model becomes promotable
    # --------------------------------------------------

    if (
        (result.get("applied") and result.get("to_state") == "PROMOTABLE")
        or (decision_type == "MANUAL_OVERRIDE" and current_state == "PROMOTABLE")
    ):
        print("[Auto Executor] Model is PROMOTABLE. Executing shadow promotion...")
        exit_code = os.system("python -m scripts.promote_from_shadow")
        if exit_code != 0:
            raise RuntimeError("Shadow promotion failed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[Auto Executor] ERROR:", e)
        sys.exit(1)