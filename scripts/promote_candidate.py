from pathlib import Path
import json
import shutil
from datetime import datetime, timezone


PROD_CONFIG = Path("config/production_model.json")
CANDIDATE_DIR = Path("artifacts/models/candidate")
MODEL_ROOT = Path("artifacts/models/lightgbm")


def parse_version(version: str):
    major, minor, patch = map(int, version.lstrip("v").split("."))
    return major, minor, patch


def next_version_from_artifacts() -> str:
    """
    Determine next version by inspecting existing model artifacts,
    not the currently active production pointer.
    """
    versions = []
    for d in MODEL_ROOT.iterdir():
        if d.is_dir() and d.name.startswith("v"):
            try:
                versions.append(parse_version(d.name))
            except Exception:
                continue

    if not versions:
        return "v1.0.0"

    major, minor, patch = max(versions)
    return f"v{major}.{minor + 1}.0"


def main() -> None:
    # --------------------------------------------------
    # 1. Load candidate evaluation
    # --------------------------------------------------

    eval_path = CANDIDATE_DIR / "evaluation.json"

    if not eval_path.exists():
        raise FileNotFoundError("Missing candidate evaluation.json")

    evaluation = json.loads(eval_path.read_text())

    if evaluation.get("decision") != "PROMOTE":
        print("Candidate not approved for promotion.")
        print(evaluation.get("reasons"))
        return

    # --------------------------------------------------
    # 2. Determine next version (artifact-based)
    # --------------------------------------------------

    new_version = next_version_from_artifacts()
    new_model_dir = MODEL_ROOT / new_version

    if new_model_dir.exists():
        raise RuntimeError(
            f"Target model version already exists: {new_version}"
        )

    # --------------------------------------------------
    # 3. Promote candidate (immutable copy)
    # --------------------------------------------------

    shutil.copytree(CANDIDATE_DIR, new_model_dir)

    candidate_metadata_path = new_model_dir / "metadata.json"
    if not candidate_metadata_path.exists():
        raise RuntimeError("Candidate metadata.json is missing, cannot promote.")

    candidate_metadata = json.loads(candidate_metadata_path.read_text())

    feature_contract = candidate_metadata.get("feature_contract")
    if not feature_contract:
        raise RuntimeError(
            "Promotion blocked: missing feature_contract in candidate metadata."
        )

    # Remove evaluation-only files from production artifact
    eval_copy = new_model_dir / "evaluation.json"
    if eval_copy.exists():
        eval_copy.unlink()

    # --------------------------------------------------
    # 4. Write production metadata
    # --------------------------------------------------

    metadata = {
        "model_name": "lightgbm",
        "model_version": new_version,
        "feature_contract": candidate_metadata["feature_contract"],
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promotion_reason": evaluation.get("reasons"),
    }

    with (new_model_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    # --------------------------------------------------
    # 5. Update production pointer
    # --------------------------------------------------

    PROD_CONFIG.write_text(
        json.dumps(
            {
                "model_name": "lightgbm",
                "model_version": new_version,
            },
            indent=2,
        )
    )

    # --------------------------------------------------
    # 6. Invalidate candidate after promotion
    # --------------------------------------------------

    shutil.rmtree(CANDIDATE_DIR)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model promoted successfully → {new_version}")


if __name__ == "__main__":
    main()