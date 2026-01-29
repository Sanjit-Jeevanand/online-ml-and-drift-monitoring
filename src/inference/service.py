from fastapi import FastAPI, HTTPException
from typing import Dict
import time
import json
import sys
import uuid
from fastapi.responses import JSONResponse

from src.inference.predictor import Predictor
from src.inference.schemas import PredictRequest, PredictResponse

from src.monitoring.builders import build_inference_log_event
from src.monitoring.sinks import StdoutSink, JsonlFileSink, MultiSink
from src.features.contracts import ALL_FEATURES
from pathlib import Path

PROD_CONFIG = Path("config/production_model.json")
SHADOW_CONFIG = Path("config/shadow_model.json")


# ============================================================
# Globals
# ============================================================


# Global sink holder
LOG_SINK = None


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="Online ML Inference Service",
    description="Serves real-time risk predictions.",
    version="0.1.0",
)


# ============================================================
# Startup hook
# ============================================================

@app.on_event("startup")
def load_predictors() -> None:
    global LOG_SINK

    prod_cfg = json.loads(PROD_CONFIG.read_text())

    app.state.prod_predictor = Predictor(
        model_name=prod_cfg["model_name"],
        model_version=prod_cfg["model_version"],
    )

    if SHADOW_CONFIG.exists():
        shadow_cfg = json.loads(SHADOW_CONFIG.read_text())
        if shadow_cfg.get("source") == "candidate":
            # Load candidate model from artifacts/models/candidate without model_name/version
            app.state.shadow_predictor = Predictor(
                artifact_path=Path("artifacts/models/candidate")
            )
        elif "model_name" in shadow_cfg and "model_version" in shadow_cfg:
            app.state.shadow_predictor = Predictor(
                model_name=shadow_cfg["model_name"],
                model_version=shadow_cfg["model_version"],
            )
        else:
            app.state.shadow_predictor = None
    else:
        app.state.shadow_predictor = None

    LOG_SINK = MultiSink(
        sinks=[
            StdoutSink(),
            JsonlFileSink(Path("logs/inference.jsonl")),
        ]
    )






# ============================================================
# Routes
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "inference",
        "version": "aws-debug-2026-01-27-20-25"
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:

    request_id = str(uuid.uuid4())

    request_start = time.perf_counter()

    prod_predictor: Predictor = app.state.prod_predictor
    shadow_predictor: Predictor = app.state.shadow_predictor

    expected_features = ALL_FEATURES

    status = "success"
    inference_ms = None
    latency_ms = None

    try:
        # -----------------------------
        # Production inference timing
        # -----------------------------

        inference_start = time.perf_counter()
        proba = prod_predictor.predict_proba(request.features)
        inference_end = time.perf_counter()

        inference_ms = (inference_end - inference_start) * 1000.0

        # -----------------------------
        # Build production response
        # -----------------------------

        response = PredictResponse(
            model_name=prod_predictor.model_name,
            model_version=prod_predictor.model_version,
            predicted_probability=proba,
        )

        # Emit production log event
        LOG_SINK.emit(build_inference_log_event(
            request_id=request_id,
            model_name=prod_predictor.model_name,
            model_version=prod_predictor.model_version,
            raw_features=request.features,
            expected_features=expected_features,
            predicted_probability=proba,
            latency_ms=(time.perf_counter() - request_start) * 1000.0,
            inference_ms=inference_ms,
            status="success",
        ).to_dict())

        # === Shadow inference (non-blocking) ===
        if shadow_predictor is not None:
            try:
                shadow_inference_start = time.perf_counter()
                shadow_proba = shadow_predictor.predict_proba(request.features)
                shadow_inference_end = time.perf_counter()
                shadow_inference_ms = (shadow_inference_end - shadow_inference_start) * 1000.0

                LOG_SINK.emit(build_inference_log_event(
                    request_id=request_id,
                    model_name=shadow_predictor.model_name,
                    model_version=shadow_predictor.model_version,
                    raw_features=request.features,
                    expected_features=expected_features,
                    predicted_probability=shadow_proba,
                    latency_ms=(time.perf_counter() - request_start) * 1000.0,
                    inference_ms=shadow_inference_ms,
                    status="shadow",
                ).to_dict())
            except Exception:
                # Shadow inference errors should never affect client or main response
                pass

        return response

    # ---------------------------------------------------------
    # Client / contract errors
    # ---------------------------------------------------------

    except ValueError as e:
        status = "client_error"

        error_payload = {
            "error_type": "validation_error",
            "message": str(e),
            "model_name": prod_predictor.model_name,
            "model_version": prod_predictor.model_version,
        }

        LOG_SINK.emit(build_inference_log_event(
            request_id=request_id,
            model_name=prod_predictor.model_name,
            model_version=prod_predictor.model_version,
            raw_features=request.features,
            expected_features=expected_features,
            predicted_probability=None,
            latency_ms=(time.perf_counter() - request_start) * 1000.0,
            inference_ms=None,
            status="client_error",
            error=e,
        ).to_dict())

        return JSONResponse(
            status_code=400,
            content=error_payload,
        )

    # ---------------------------------------------------------
    # Server / inference errors
    # ---------------------------------------------------------

    except Exception as e:
        status = "server_error"

        error_payload = {
            "error_type": "inference_error",
            "message": "Internal inference error.",
            "model_name": prod_predictor.model_name,
            "model_version": prod_predictor.model_version,
        }

        LOG_SINK.emit(build_inference_log_event(
            request_id=request_id,
            model_name=prod_predictor.model_name,
            model_version=prod_predictor.model_version,
            raw_features=request.features,
            expected_features=expected_features,
            predicted_probability=None,
            latency_ms=(time.perf_counter() - request_start) * 1000.0,
            inference_ms=None,
            status="server_error",
            error=e,
        ).to_dict())

        return JSONResponse(
            status_code=500,
            content=error_payload,
        )