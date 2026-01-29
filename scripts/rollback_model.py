from pathlib import Path
import json
import sys
from datetime import datetime


PROD_CONFIG = Path("config/production_model.json")
MODEL_ROOT = Path("artifacts/models/lightgbm")


def parse_version(v: str):
    return tuple(map(int, v.lstrip("v").split(".")))


def main() -> None:
    # --------------------------------------------------
    # 1. Load current production config
    # --------------------------------------------------

    if not PROD_CONFIG.exists():
        raise FileNotFoundError("Missing production_model.json")

    prod_cfg = json.loads(PROD_CONFIG.read_text())
    current_version = prod_cfg["model_version"]

    # --------------------------------------------------
    # 2. Discover available versions
    # --------------------------------------------------

    versions = sorted(
        [d.name for d in MODEL_ROOT.iterdir() if d.is_dir()],
        key=parse_version,
    )

    if current_version not in versions:
        raise RuntimeError(
            f"Current production version {current_version} not found on disk"
        )

    current_idx = versions.index(current_version)

    if current_idx == 0:
        print("No previous version available — rollback aborted.")
        return

    rollback_version = versions[current_idx - 1]

    # --------------------------------------------------
    # 3. Perform rollback (pointer update only)
    # --------------------------------------------------

    prod_cfg["model_version"] = rollback_version
    prod_cfg["rolled_back_at"] = datetime.utcnow().isoformat()
    prod_cfg["rollback_from"] = current_version

    PROD_CONFIG.write_text(json.dumps(prod_cfg, indent=2))

    print(
        f"Rollback successful: {current_version} → {rollback_version}"
    )


if __name__ == "__main__":
    main()