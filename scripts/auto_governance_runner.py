

"""
Automatic Governance Runner

This script is intended to be executed on a schedule (cron / cloud scheduler).
It evaluates the current governance state and executes safe, deterministic
actions such as state advancement, promotion, or rollback.

It never trains models and never makes heuristic decisions.
All intelligence lives in the governance controller and downstream executors.
"""

from datetime import datetime, timezone

from src.governance.controller import run_governance_controller
from src.governance.state_transition import apply_state_transition

from scripts.promote_from_shadow import main as promote_from_shadow
from scripts.rollback_model import main as rollback_model


def main() -> None:
    print(f"[Auto Governance] Run started at {datetime.now(timezone.utc).isoformat()}")

    decision = run_governance_controller()

    decision_type = decision.get("decision")
    current_state = decision.get("current_state")
    next_state = decision.get("next_state")

    print(f"[Auto Governance] Decision: {decision_type}")
    print(f"[Auto Governance] State: {current_state} → {next_state}")

    # --------------------------------------------------
    # 1. Automatic state advancement (safe, idempotent)
    # --------------------------------------------------
    if decision_type == "AUTO_ADVANCE":
        print("[Auto Governance] Applying automatic state transition...")
        result = apply_state_transition()
        print("[Auto Governance] Transition result:")
        print(result)
        return

    # --------------------------------------------------
    # 2. Promotion path
    # --------------------------------------------------
    if decision_type in {"PROMOTABLE", "MANUAL_OVERRIDE"} and current_state == "PROMOTABLE":
        print("[Auto Governance] Model is PROMOTABLE. Executing promotion...")
        try:
            promote_from_shadow()
        except Exception as e:
            print("[Auto Governance] ERROR: Promotion failed.")
            raise
        return

    # --------------------------------------------------
    # 3. Rollback path
    # --------------------------------------------------
    if decision_type == "ROLLBACK":
        print("[Auto Governance] Rollback triggered. Executing rollback...")
        try:
            rollback_model()
        except Exception as e:
            print("[Auto Governance] ERROR: Rollback failed.")
            raise
        return

    # --------------------------------------------------
    # 4. No-op
    # --------------------------------------------------
    print("[Auto Governance] No executable action. Exiting.")


if __name__ == "__main__":
    main()