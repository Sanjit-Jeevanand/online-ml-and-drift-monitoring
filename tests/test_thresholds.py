from src.monitoring.thresholds import classify_ks, classify_psi

def test_ks_thresholds():
    assert classify_ks(0.05) == "none"
    assert classify_ks(0.15) == "low"
    assert classify_ks(0.25) == "high"

def test_psi_thresholds():
    assert classify_psi(0.05) == "none"
    assert classify_psi(0.15) == "low"
    assert classify_psi(0.30) == "high"