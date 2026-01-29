from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
import json


# ============================================================
# Paths
# ============================================================

GOVERNANCE_DIR = Path("artifacts/governance")
STATE_PATH = GOVERNANCE_DIR / "state.json"
DECISION_PATH = GOVERNANCE_DIR / "decision.json"
OVERRIDE_PATH = GOVERNANCE_DIR / "manual_override.json"
SHADOW_ANALYSIS_PATH = Path("artifacts/shadow/analysis.json")


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


def _load_manual_override() -> Dict[str, Any] | None:
    if not OVERRIDE_PATH.exists():
        return None

    override = json.loads(OVERRIDE_PATH.read_text())

    if not override.get("enabled", False):
        return None

    expires_at = override.get("expires_at")
    if not expires_at:
        return None

    expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    if datetime.now(timezone.utc) > expires:
        return None

    return override


# ============================================================
# Policy evaluation
# ============================================================

def evaluate_shadow_policy(
    shadow: Dict[str, Any],
    *,
    min_requests: int = 500,
    max_p95_pred_delta: float = 0.05,
    max_p95_shadow_latency_ms: float = 20.0,
    max_shadow_error_rate: float = 0.0,
) -> List[str]:
    """
    Returns a list of blocking reasons.
    Empty list = policy passed.
    """

    reasons = []

    counts = shadow.get("counts", {})
    n_requests = counts.get("paired_requests", 0)

    if n_requests < min_requests:
        reasons.append(
            f"Insufficient shadow traffic ({n_requests} < {min_requests})"
        )

    pred = shadow.get("prediction_comparison", {})
    p95_abs_delta = pred.get("p95_abs_delta")

    if p95_abs_delta is not None and p95_abs_delta > max_p95_pred_delta:
        reasons.append(
            f"Prediction divergence too high (p95_abs_delta={p95_abs_delta:.4f})"
        )

    latency = shadow.get("latency_comparison", {})
    p95_shadow_ms = latency.get("p95_shadow_ms")

    if p95_shadow_ms is not None and p95_shadow_ms > max_p95_shadow_latency_ms:
        reasons.append(
            f"Shadow latency too high (p95_shadow_ms={p95_shadow_ms:.2f})"
        )

    errors = shadow.get("error_rates", {})
    shadow_error_rate = errors.get("shadow")

    if shadow_error_rate is not None and shadow_error_rate > max_shadow_error_rate:
        reasons.append(
            f"Shadow error rate too high ({shadow_error_rate})"
        )

    return reasons


# ============================================================
# State transition logic
# ============================================================

def decide_next_state(
    *,
    current_state: str,
    shadow_analysis: Dict[str, Any],
) -> Dict[str, Any]:

    reasons: List[str] = []
    allowed_transitions: List[str] = []
    blocked_transitions: List[str] = []

    # --------------------------------------------------------
    # Only SHADOWING → PROMOTABLE is automated (for now)
    # --------------------------------------------------------

    if current_state != "SHADOWING":
        return {
            "decision": "NO_OP",
            "current_state": current_state,
            "next_state": current_state,
            "allowed_transitions": [],
            "blocked_transitions": [],
            "reasons": [
                f"No automated transitions defined for state '{current_state}'"
            ],
        }

    # --------------------------------------------------------
    # Evaluate shadow policy
    # --------------------------------------------------------

    reasons = evaluate_shadow_policy(shadow_analysis)

    if reasons:
        blocked_transitions.append("PROMOTABLE")
        return {
            "decision": "BLOCK",
            "current_state": current_state,
            "next_state": "SHADOWING",
            "allowed_transitions": [],
            "blocked_transitions": blocked_transitions,
            "reasons": reasons,
        }

    allowed_transitions.append("PROMOTABLE")

    return {
        "decision": "AUTO_ADVANCE",
        "current_state": current_state,
        "next_state": "PROMOTABLE",
        "allowed_transitions": allowed_transitions,
        "blocked_transitions": [],
        "reasons": [
            "All shadow policy checks passed",
            "Automatic promotion conditions satisfied",
        ],
    }


# ============================================================
# Main entry point
# ============================================================

def run_governance_controller() -> Dict[str, Any]:
    """
    Pure governance decision engine.
    Reads artifacts, emits a decision artifact.
    """

    # --------------------------------------------------------
    # Load required artifacts
    # --------------------------------------------------------

    state = _load_json(STATE_PATH)
    shadow = _load_json(SHADOW_ANALYSIS_PATH)

    current_state = state.get("state")

    if not current_state:
        raise ValueError("Governance state missing 'state' field.")

    # --------------------------------------------------------
    # Decide next transition
    # --------------------------------------------------------

    decision_core = decide_next_state(
        current_state=current_state,
        shadow_analysis=shadow,
    )

    override = _load_manual_override()

    if override:
        if (
            override.get("model_name") == state.get("model_name")
            and override.get("model_version") == state.get("model_version")
        ):
            decision = {
                "decision": "MANUAL_OVERRIDE",
                "current_state": current_state,
                "next_state": override["requested_transition"],
                "allowed_transitions": [override["requested_transition"]],
                "blocked_transitions": [],
                "reasons": [
                    "Manual override applied",
                    f"Approved by: {override.get('approved_by')}",
                    override.get("reason"),
                ],
                "model_name": state.get("model_name"),
                "model_version": state.get("model_version"),
                "evaluated_at": utc_now(),
            }
            _write_json(DECISION_PATH, decision)
            return decision

    decision = {
        **decision_core,
        "model_name": state.get("model_name"),
        "model_version": state.get("model_version"),
        "evaluated_at": utc_now(),
    }

    # --------------------------------------------------------
    # Persist decision
    # --------------------------------------------------------

    _write_json(DECISION_PATH, decision)

    return decision