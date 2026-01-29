from pathlib import Path
import json
import numpy as np
from datetime import datetime, timezone

from src.models.artifacts import load_model
from src.models.evaluation import evaluate_binary_classifier


PROD_MODEL_DIR = Path("artifacts/models/lightgbm/v1.1.0")
CANDIDATE_DIR = Path("artifacts/models/candidate")

X_VAL = Path("artifacts/features/X_val.npy")
Y_VAL = Path("artifacts/labels/y_val.npy")


def main() -> None:
    # --------------------------------------------------
    # 1. Load validation data
    # --------------------------------------------------

    X_val = np.load(X_VAL, allow_pickle=True)
    y_val = np.load(Y_VAL, allow_pickle=True).astype(int)

    # --------------------------------------------------
    # 2. Load production model
    # --------------------------------------------------

    prod_model = load_model(PROD_MODEL_DIR / "model.joblib")

    # --------------------------------------------------
    # 3. Load candidate model
    # --------------------------------------------------

    candidate_model = load_model(CANDIDATE_DIR / "model.joblib")

    # --------------------------------------------------
    # 4. Evaluate both models
    # --------------------------------------------------

    prod_metrics = evaluate_binary_classifier(
        model=prod_model,
        X=X_val,
        y=y_val,
    )

    cand_metrics = evaluate_binary_classifier(
        model=candidate_model,
        X=X_val,
        y=y_val,
    )

    # --------------------------------------------------
    # 5. Compare metrics
    # --------------------------------------------------

    deltas = {
        k: cand_metrics[k] - prod_metrics[k]
        for k in prod_metrics
    }

    promote = (
        cand_metrics["roc_auc"] >= prod_metrics["roc_auc"]
        and cand_metrics["pr_auc"] >= prod_metrics["pr_auc"]
        and cand_metrics["brier_score"] <= prod_metrics["brier_score"]
    )

    reasons = []

    for k, delta in deltas.items():
        direction = "improved" if delta > 0 else "degraded"
        reasons.append(f"{k} {direction} by {delta:.6f}")

    decision = {
        "decision": "PROMOTE" if promote else "REJECT",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "production": prod_metrics,
            "candidate": cand_metrics,
            "delta": deltas,
        },
        "reasons": reasons,
    }

    # --------------------------------------------------
    # 6. Persist evaluation
    # --------------------------------------------------

    output_path = CANDIDATE_DIR / "evaluation.json"
    with output_path.open("w") as f:
        json.dump(decision, f, indent=2)

    print("Candidate evaluation complete:")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()