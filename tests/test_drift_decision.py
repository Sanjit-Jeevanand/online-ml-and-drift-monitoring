from src.monitoring.drift_decision import decide_drift_action

def test_prediction_drift_triggers_retrain():
    decision = decide_drift_action(
        ks_metrics={},
        psi_metrics={},
        prediction_metrics={"mean_shift": 0.5, "p95_shift": 0.3},
        baseline_prediction_stats={"std": 0.1, "quantiles": {"p95": 0.6}},
        baseline_volume=1000,
        current_volume=1000,
    )

    assert decision["decision"] == "RETRAIN_REQUIRED"