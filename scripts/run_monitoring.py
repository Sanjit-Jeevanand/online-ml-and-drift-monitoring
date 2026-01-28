from pathlib import Path
from datetime import datetime, timedelta
import json

from src.monitoring.aggregation import read_inference_logs
from src.monitoring.snapshots import build_snapshot, save_snapshot
from src.monitoring.drift_metrics import (
    ks_drift,
    psi_drift,
    prediction_drift,
    volume_drift,
)
from src.monitoring.drift_decision import decide_drift_action


LOG_PATH = Path("logs/inference.jsonl")
BASELINE_SNAPSHOT_PATH = Path("snapshots/baseline.json")
CURRENT_SNAPSHOT_PATH = Path("snapshots/current.json")
DECISION_PATH = Path("artifacts/drift/decision.json")

WINDOW_MINUTES = 10


def main() -> None:
    now = datetime.utcnow()
    window_start = now - timedelta(minutes=WINDOW_MINUTES)

    # --------------------------------------------------
    # 1. Read recent inference logs
    # --------------------------------------------------

    records = read_inference_logs(
        LOG_PATH,
        since_minutes=WINDOW_MINUTES,
    )

    if not records:
        print("No inference records found — skipping monitoring.")
        return

    # --------------------------------------------------
    # 2. Build current snapshot
    # --------------------------------------------------

    current_snapshot = build_snapshot(
        records,
        model_name="lightgbm",
        model_version="v1.1.0",
        feature_version="v1",
        window_start=window_start,
        window_end=now,
    )

    save_snapshot(current_snapshot, CURRENT_SNAPSHOT_PATH)

    # --------------------------------------------------
    # 3. Load baseline snapshot
    # --------------------------------------------------

    baseline_snapshot = json.loads(
        BASELINE_SNAPSHOT_PATH.read_text()
    )

    # --------------------------------------------------
    # 4. Compute drift metrics
    # --------------------------------------------------

    ks = ks_drift(baseline_snapshot, current_snapshot)
    psi = psi_drift(baseline_snapshot, current_snapshot)
    pred = prediction_drift(baseline_snapshot, current_snapshot)
    vol = volume_drift(baseline_snapshot, current_snapshot)

    # --------------------------------------------------
    # 5. Decide action
    # --------------------------------------------------

    decision = decide_drift_action(
        ks_metrics=ks,
        psi_metrics=psi,
        prediction_metrics=pred,
        baseline_prediction_stats=baseline_snapshot["predictions"],
        baseline_volume=baseline_snapshot["volume"]["n_requests"],
        current_volume=current_snapshot["volume"]["n_requests"],
    )

    DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_PATH.open("w") as f:
        json.dump(decision, f, indent=2)

    print("Monitoring decision:")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()