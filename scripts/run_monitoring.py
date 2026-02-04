from pathlib import Path
from datetime import datetime, timedelta, timezone
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
BASELINE_SNAPSHOT_PATH = Path("artifacts/snapshots/baseline.json")
CURRENT_SNAPSHOT_PATH = Path("artifacts/snapshots/current.json")
DECISION_PATH = Path("artifacts/drift/decision.json")
PRODUCTION_MODEL_PATH = Path("config/production_model.json")

WINDOW_MINUTES = 30


def main() -> None:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=WINDOW_MINUTES)

    # --------------------------------------------------
    # 1. Read recent inference logs
    # --------------------------------------------------

    records = read_inference_logs(
        LOG_PATH,
        since_minutes=WINDOW_MINUTES,
    )

    print(f"[Monitoring] Read {len(records)} inference records")

    if not records:
        print("[Monitoring] No inference records found — skipping monitoring.")
        return

    # --------------------------------------------------
    # 2. Build current snapshot
    # --------------------------------------------------

    production_model = json.loads(PRODUCTION_MODEL_PATH.read_text())
    active_model_version = production_model["model_version"]

    current_snapshot = build_snapshot(
        records,
        model_name=production_model["model_name"],
        model_version=active_model_version,
        feature_version="v1",
        window_start=window_start,
        window_end=now,
    )

    save_snapshot(current_snapshot, CURRENT_SNAPSHOT_PATH)

    # --------------------------------------------------
    # 3. Load baseline snapshot
    # --------------------------------------------------

    if not BASELINE_SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            f"Baseline snapshot missing: {BASELINE_SNAPSHOT_PATH}"
        )

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
        baseline_prediction_stats=baseline_snapshot.get("predictions"),
        baseline_volume=baseline_snapshot.get("volume", {}).get("n_requests"),
        current_volume=current_snapshot.get("volume", {}).get("n_requests"),
    )

    DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_PATH.open("w") as f:
        json.dump(decision, f, indent=2)

    print("Monitoring decision:")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()