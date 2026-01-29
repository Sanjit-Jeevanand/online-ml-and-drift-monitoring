from pathlib import Path
import json
import sys
from datetime import datetime, timezone
from typing import Dict, Any

from src.monitoring.shadow_analysis import run_shadow_analysis
from src.models.registry import register_model


# ============================================================
# Paths
# ============================================================

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
CANDIDATE_DIR = MODELS_DIR / "candidate"
DECISIONS_DIR = ARTIFACTS_DIR / "shadow"
DECISIONS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Promotion thresholds (hard gates)
# ============================================================

MIN_REQUESTS = 500
MAX_P95_PRED_DELTA = 0.02
MAX_P95_SHADOW_LATENCY_MS = 15.0
MAX_SHADOW_ERROR_RATE = 0.0


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_candidate_metadata() -> Dict[str, Any]:
    path = CANDIDATE_DIR / "metadata.json"
    if not path.exists():
        raise FileNotFoundError("Candidate metadata.json not found.")
    return json.loads(path.read_text())


def invalidate_candidate() -> None:
    if not CANDIDATE_DIR.exists():
        return

    archived = MODELS_DIR / f"candidate_archived_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    CANDIDATE_DIR.rename(archived)


# ============================================================
# Main logic
# ============================================================

def main() -> None:
    print("[Shadow Promotion] Running shadow analysis...")

    shadow_metrics = run_shadow_analysis()

    # --------------------------------------------------------
    # Load candidate metadata
    # --------------------------------------------------------

    candidate_meta = load_candidate_metadata()

    model_name = candidate_meta.get("model_name", "lightgbm")
    candidate_version = candidate_meta.get("candidate_version")

    if not candidate_version:
        print("[Shadow Promotion] ERROR: candidate_version missing in metadata.")
        sys.exit(1)

    # --------------------------------------------------------
    # Hard safety gates (derived ONLY from shadow_analysis)
    # --------------------------------------------------------

    reasons = []

    n_requests = shadow_metrics["counts"]["paired_requests"]
    if n_requests < MIN_REQUESTS:
        reasons.append(
            f"Insufficient shadow traffic ({n_requests} < {MIN_REQUESTS})"
        )

    p95_pred = shadow_metrics["prediction_comparison"]["p95_abs_delta"]
    if p95_pred > MAX_P95_PRED_DELTA:
        reasons.append(
            f"Prediction divergence too high (p95_abs_delta={p95_pred:.4f})"
        )

    p95_shadow_latency = shadow_metrics["latency_comparison"]["p95_shadow_ms"]
    if p95_shadow_latency > MAX_P95_SHADOW_LATENCY_MS:
        reasons.append(
            f"Shadow latency too high (p95_shadow_ms={p95_shadow_latency:.2f})"
        )

    shadow_error_rate = shadow_metrics["error_rates"]["shadow"]
    if shadow_error_rate > MAX_SHADOW_ERROR_RATE:
        reasons.append(
            f"Shadow error rate > 0 ({shadow_error_rate})"
        )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decision = "PROMOTE" if not reasons else "DO_NOT_PROMOTE"

    decision_record = {
        "decision": decision,
        "candidate_version": candidate_version,
        "model_name": model_name,
        "timestamp_utc": utc_now(),
        "shadow_summary": {
            "paired_requests": n_requests,
            "p95_abs_prediction_delta": p95_pred,
            "p95_shadow_latency_ms": p95_shadow_latency,
            "shadow_error_rate": shadow_error_rate,
        },
        "reasons": reasons,
    }

    decision_path = DECISIONS_DIR / "shadow_promotion_decision.json"
    decision_path.write_text(json.dumps(decision_record, indent=2))

    # --------------------------------------------------------
    # Act on decision
    # --------------------------------------------------------

    if decision != "PROMOTE":
        print("[Shadow Promotion] Promotion blocked.")
        for r in reasons:
            print(f" - {r}")
        sys.exit(0)

    print("[Shadow Promotion] Promotion approved.")

    # --------------------------------------------------------
    # Promote candidate → registry
    # --------------------------------------------------------

    model_pointer = json.loads((CANDIDATE_DIR / "model_pointer.json").read_text())
    promoted_version = model_pointer["next_version"]

    print(f"[Shadow Promotion] Promoting candidate → {promoted_version}")

    register_model(
        model_name=model_name,
        version=promoted_version,
        model_path=CANDIDATE_DIR / "model.joblib",
        preprocessor_path=CANDIDATE_DIR / "preprocessor.joblib",
        metrics_path=CANDIDATE_DIR / "metrics.json",
        calibration_path=CANDIDATE_DIR / "calibration.npz",
        metadata=candidate_meta,
    )

    invalidate_candidate()

    print("[Shadow Promotion] Promotion complete.")
    print(f"[Shadow Promotion] New production version: {promoted_version}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()