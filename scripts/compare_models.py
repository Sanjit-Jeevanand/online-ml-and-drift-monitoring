from pathlib import Path
import json


# ============================================================
# Paths
# ============================================================

ARTIFACTS_DIR = Path("artifacts/models")

BASELINE_DIR = ARTIFACTS_DIR / "baseline"
LGBM_DIR = ARTIFACTS_DIR / "lightgbm"
XGB_DIR = ARTIFACTS_DIR / "xgboost"


# ============================================================
# Utilities
# ============================================================

def load_metrics(path: Path) -> dict:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with open(metrics_path, "r") as f:
        return json.load(f)


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.4f}")


# ============================================================
# Promotion logic
# ============================================================

def should_promote(
    baseline: dict,
    candidate: dict,
    *,
    min_auc_gain: float = 0.005,
    max_brier_regression: float = 0.005,
) -> bool:

    auc_gain = candidate["roc_auc"] - baseline["roc_auc"]
    pr_gain = candidate["pr_auc"] - baseline["pr_auc"]
    brier_change = candidate["brier_score"] - baseline["brier_score"]

    if auc_gain < min_auc_gain:
        return False

    if pr_gain <= 0:
        return False

    if brier_change > max_brier_regression:
        return False

    return True


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    print("Loading model metrics...")

    baseline_metrics = load_metrics(BASELINE_DIR)
    lgbm_metrics = load_metrics(LGBM_DIR)

    print_metrics("Baseline (Logistic Regression)", baseline_metrics)
    print_metrics("LightGBM (Tuned)", lgbm_metrics)

    promote_lgbm = should_promote(
        baseline_metrics,
        lgbm_metrics,
    )

    print("\nPromotion decision")
    print("------------------")

    if promote_lgbm:
        print("LightGBM SHOULD be promoted to production.")
    else:
        print("LightGBM should NOT be promoted.")

    if XGB_DIR.exists():
        try:
            xgb_metrics = load_metrics(XGB_DIR)
            print_metrics("XGBoost", xgb_metrics)

            promote_xgb = should_promote(
                baseline_metrics,
                xgb_metrics,
            )

            if promote_xgb:
                print("\nXGBoost also meets promotion criteria.")
        except FileNotFoundError:
            pass


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()