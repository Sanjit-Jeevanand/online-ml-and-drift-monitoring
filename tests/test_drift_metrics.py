from src.monitoring.drift_metrics import ks_drift

def test_ks_detects_shift():
    baseline = {
        "features": {"LIMIT_BAL": [100, 200, 300]}
    }
    current = {
        "features": {"LIMIT_BAL": [1000, 2000, 3000]}
    }

    ks = ks_drift(baseline, current)
    assert ks["LIMIT_BAL"] > 0.5