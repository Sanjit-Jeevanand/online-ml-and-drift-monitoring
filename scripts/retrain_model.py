from pathlib import Path
import json
import sys
from datetime import datetime, timezone

# --------------------------------------------------
# Paths
# --------------------------------------------------

DECISION_PATH = Path("artifacts/drift/decision.json")
RETRAIN_SIGNAL_PATH = Path("artifacts/drift/retrain_signal.json")

# --------------------------------------------------
# Allowed retraining decisions
# --------------------------------------------------

RETRAIN_DECISIONS = {
    "RETRAIN_RECOMMENDED",
    "RETRAIN_REQUIRED",
}

# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    print("[Retrain Orchestrator] Starting retrain decision check...")

    if not DECISION_PATH.exists():
        print("[Retrain Orchestrator] No drift decision found. Exiting.")
        sys.exit(0)

    with DECISION_PATH.open("r") as f:
        decision_payload = json.load(f)

    decision = decision_payload.get("decision")
    severity = decision_payload.get("severity")
    reasons = decision_payload.get("reasons", [])

    print(f"[Retrain Orchestrator] Drift decision: {decision}")
    print(f"[Retrain Orchestrator] Severity: {severity}")

    if decision not in RETRAIN_DECISIONS:
        print("[Retrain Orchestrator] Retraining NOT permitted. Monitoring only.")
        sys.exit(0)

    # --------------------------------------------------
    # Emit retraining signal (no training yet)
    # --------------------------------------------------

    signal = {
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "severity": severity,
        "reasons": reasons,
    }

    RETRAIN_SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RETRAIN_SIGNAL_PATH.open("w") as f:
        json.dump(signal, f, indent=2)

    print("[Retrain Orchestrator] Retraining signal emitted.")
    print(f"[Retrain Orchestrator] Signal path: {RETRAIN_SIGNAL_PATH}")

    # Exit code 1 intentionally signals retraining should proceed
    sys.exit(1)


if __name__ == "__main__":
    main()
