from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import uuid
import time

import joblib
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

LOG_PATH = Path("logs/inference.jsonl")

FEATURES_DIR = Path("artifacts/features")

PROD_CONFIG_PATH = Path("config/production_model.json")
MODELS_DIR = Path("artifacts/models")

SHADOW_DIR = Path("artifacts/models/candidate")

N_REQUESTS = 4500
WINDOW_MINUTES = 10


# --------------------------------------------------
# Helpers
# --------------------------------------------------



# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    print("[Simulator] Running real model inference")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load production model
    # --------------------------------------------------

    prod_cfg = json.loads(PROD_CONFIG_PATH.read_text())

    prod_dir = (
        MODELS_DIR
        / prod_cfg["model_name"]
        / prod_cfg["model_version"]
    )

    prod_model = joblib.load(prod_dir / "model.joblib")

    print(f"[Simulator] Loaded production model {prod_cfg}")

    # --------------------------------------------------
    # 2. Load shadow model
    # --------------------------------------------------

    shadow_model = joblib.load(SHADOW_DIR / "model.joblib")

    print("[Simulator] Loaded shadow candidate model")

    # --------------------------------------------------
    # 3. Load features and metadata
    # --------------------------------------------------

    X = np.load(FEATURES_DIR / "X_val.npy")

    feature_metadata = json.loads((FEATURES_DIR / "feature_metadata.json").read_text())

    # feature_metadata is a dict, not a list
    # Expected keys: "feature_names", "feature_types", "version"
    feature_names = feature_metadata["feature_names"]

    # --------------------------------------------------
    # SIMPLE FEATURE DRIFT (MEAN SHIFT + NOISE)
    # --------------------------------------------------
    # Apply moderate drift to shadow features so KS is non-zero
    # but still realistic and stable for demo purposes

    rng = np.random.default_rng(seed=42)

    X_shadow = X.copy()

    # drift_strength = 0.1   # mean shift
    # noise_scale = 0.05      # added variance

    # X_shadow = (
    #     X_shadow
    #     + drift_strength
    #     + rng.normal(0.0, noise_scale, size=X_shadow.shape)
    # )

    n_available = X.shape[0]
    n_requests = min(N_REQUESTS, n_available)

    if n_requests < N_REQUESTS:
        print(
            f"[Simulator] Warning: Requested {N_REQUESTS} samples but only "
            f"{n_available} available. Using {n_requests}."
        )

    # Deterministic slice for reproducibility
    X = X[:n_requests]
    X_shadow = X_shadow[:n_requests]

    # --------------------------------------------------
    # 4. Generate logs
    # --------------------------------------------------

    now = datetime.now(timezone.utc)

    with LOG_PATH.open("a") as f:
        for i, row in enumerate(X):
            window_start = datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MINUTES)

            ts = window_start + timedelta(
                seconds=(WINDOW_MINUTES * 60 * i) / n_requests
            )

            request_id = str(uuid.uuid4())

            # ---- Production inference
            # Real model forward pass (not cached or replayed)
            t0 = time.perf_counter()
            prod_pred = float(
                prod_model.predict_proba(row.reshape(1, -1))[0, 1]
            )
            prod_latency = (time.perf_counter() - t0) * 1000  # ms

            # ---- Shadow inference
            # Real model forward pass (not cached or replayed)
            shadow_row = X_shadow[i]
            t1 = time.perf_counter()
            shadow_pred = float(
                shadow_model.predict_proba(shadow_row.reshape(1, -1))[0, 1]
            )
            shadow_latency = (time.perf_counter() - t1) * 1000  # ms

            # Log drifted features using true feature names for drift detection
            feature_payload = {
                feature_names[j]: float(val) for j, val in enumerate(shadow_row)
            }

            record = {
                "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
                "request_id": request_id,

                # Production inference (baseline)
                "status": "success",
                "model": {
                    "name": prod_cfg["model_name"],
                    "version": prod_cfg["model_version"],
                },
                "prediction": {
                    "predicted_probability": prod_pred
                },
                "latency_ms": prod_latency,
                "inference_ms": prod_latency,
                "error": False,

                # Feature payload (what monitoring snapshots aggregate)
                "features": {
                    "numeric": feature_payload
                },

                # Shadow inference (candidate)
                "shadow": {
                    "model": {
                        "name": "lightgbm",
                        "version": "candidate",
                    },
                    "predicted_probability": shadow_pred,
                    "latency_ms": shadow_latency,
                    "error": False,
                },
            }

            f.write(json.dumps(record) + "\n")

    print(f"[Simulator] Wrote {n_requests} paired inference records")
    print("[Simulator] Done.")


if __name__ == "__main__":
    main()