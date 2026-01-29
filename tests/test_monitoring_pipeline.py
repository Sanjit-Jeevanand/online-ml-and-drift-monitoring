import subprocess
from pathlib import Path
import json

def test_monitoring_end_to_end(tmp_path):
    # Prepare fake logs
    logs = tmp_path / "inference.jsonl"
    logs.write_text(
        '{"event_type":"inference","prediction":{"predicted_probability":0.9}}\n'
    )

    # Run monitoring
    subprocess.run(
        ["python", "-m", "scripts.run_monitoring"],
        check=True,
    )

    decision = json.loads(
        Path("artifacts/drift/decision.json").read_text()
    )

    assert "decision" in decision