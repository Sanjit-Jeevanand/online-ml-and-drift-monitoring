from pathlib import Path
import json

from src.monitoring.drift_metrics import (
    ks_drift,
    psi_drift,
    prediction_drift,
    volume_drift,
)
from src.monitoring.snapshots import load_snapshot


BASELINE_PATH = Path("snapshots/baseline.json")
CURRENT_PATH = Path("snapshots/current.json")
OUTPUT_PATH = Path("artifacts/drift/drift_metrics.json")


def main() -> None:
    baseline = load_snapshot(BASELINE_PATH)
    current = load_snapshot(CURRENT_PATH)

    drift = {
        "numeric_ks": ks_drift(baseline, current),
        "categorical_psi": psi_drift(baseline, current),
        "prediction": prediction_drift(baseline, current),
        "volume": volume_drift(baseline, current),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(drift, f, indent=2)

    print("Drift metrics computed:")
    print(json.dumps(drift, indent=2))


if __name__ == "__main__":
    main()