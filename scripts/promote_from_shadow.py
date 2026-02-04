import joblib
import numpy as np
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
ACTIVE_MODEL_PATH = MODELS_DIR / "active.json"


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


def next_production_version(model_name: str) -> str:
    prod_cfg_path = Path("config/production_model.json")

    if not prod_cfg_path.exists():
        return "v1.0.0"

    with prod_cfg_path.open("r") as f:
        cfg = json.load(f)

    current = cfg.get("model_version", "v1.0.0")
    major, minor, patch = map(int, current.lstrip("v").split("."))
    return f"v{major}.{minor}.{patch + 1}"


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

    promoted_version = next_production_version(model_name)

    candidate_meta["promoted_from_candidate"] = candidate_version
    candidate_meta["promotion_type"] = "shadow"
    candidate_meta["promoted_at"] = utc_now()
    candidate_meta["model_version"] = promoted_version

    print(f"[Shadow Promotion] Promoting candidate → {promoted_version}")

    model = joblib.load(CANDIDATE_DIR / "model.joblib")
    preprocessor = joblib.load(CANDIDATE_DIR / "preprocessor.joblib")

    with open(CANDIDATE_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)

    calibration_npz_path = CANDIDATE_DIR / "calibration.npz"
    calibration_json_path = CANDIDATE_DIR / "calibration.json"

    if calibration_npz_path.exists():
        calib_npz = np.load(calibration_npz_path)
        calibration = {
            "mean_predicted_value": calib_npz["mean_predicted_value"],
            "fraction_of_positives": calib_npz["fraction_of_positives"],
        }
    elif calibration_json_path.exists():
        with open(calibration_json_path, "r") as f:
            calibration = json.load(f)
    else:
        raise FileNotFoundError("Candidate calibration artifact not found.")

    register_model(
        model_name=model_name,
        version=promoted_version,
        model=model,
        preprocessor=preprocessor,
        metrics=metrics,
        calibration=calibration,
        metadata=candidate_meta,
    )

    active_record = {
        "model_name": model_name,
        "model_version": promoted_version,
        "promoted_from_candidate": candidate_version,
        "promotion_type": "shadow",
        "promoted_at": utc_now(),
        "feature_contract": candidate_meta.get("feature_contract"),
    }

    ACTIVE_MODEL_PATH.write_text(json.dumps(active_record, indent=2))

    # --------------------------------------------------------
    # Update production model pointer (authoritative)
    # --------------------------------------------------------

    prod_cfg_path = Path("config/production_model.json")

    prod_cfg = {
        "model_name": model_name,
        "model_version": promoted_version,
        "promotion_type": "shadow",
        "updated_at": utc_now(),
    }

    prod_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    prod_cfg_path.write_text(json.dumps(prod_cfg, indent=2))

    invalidate_candidate()

    print("[Shadow Promotion] Promotion complete.")
    print(f"[Shadow Promotion] New production version: {promoted_version}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()