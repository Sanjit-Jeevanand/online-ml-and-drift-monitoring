from fastapi import FastAPI, HTTPException
from typing import Dict
import time
import json
import sys
from fastapi.responses import JSONResponse

from src.inference.predictor import Predictor
from src.inference.schemas import PredictRequest, PredictResponse

from src.monitoring.builders import build_inference_log_event
from src.monitoring.sinks import StdoutSink, JsonlFileSink, MultiSink
from src.features.contracts import ALL_FEATURES
from pathlib import Path


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
def load_predictor() -> None:
    global LOG_SINK

    model_name = "lightgbm"
    model_version = "v1.1.0"

    app.state.predictor = Predictor(
        model_name=model_name,
        model_version=model_version,
    )

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

    request_start = time.perf_counter()

    predictor: Predictor = app.state.predictor

    expected_features = ALL_FEATURES

    status = "success"
    inference_ms = None
    latency_ms = None

    try:
        # -----------------------------
        # Inference timing
        # -----------------------------

        inference_start = time.perf_counter()
        proba = predictor.predict_proba(request.features)
        inference_end = time.perf_counter()

        inference_ms = (inference_end - inference_start) * 1000.0

        # -----------------------------
        # Build response
        # -----------------------------

        response = PredictResponse(
            model_name=predictor.model_name,
            model_version=predictor.model_version,
            predicted_probability=proba,
        )

        LOG_SINK.emit(build_inference_log_event(
            model_name=predictor.model_name,
            model_version=predictor.model_version,
            raw_features=request.features,
            expected_features=expected_features,
            predicted_probability=proba,
            latency_ms=(time.perf_counter() - request_start) * 1000.0,
            inference_ms=inference_ms,
            status="success",
        ).to_dict())

        return response

    # ---------------------------------------------------------
    # Client / contract errors
    # ---------------------------------------------------------

    except ValueError as e:
        status = "client_error"

        error_payload = {
            "error_type": "validation_error",
            "message": str(e),
            "model_name": predictor.model_name,
            "model_version": predictor.model_version,
        }

        LOG_SINK.emit(build_inference_log_event(
            model_name=predictor.model_name,
            model_version=predictor.model_version,
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
            "model_name": predictor.model_name,
            "model_version": predictor.model_version,
        }

        LOG_SINK.emit(build_inference_log_event(
            model_name=predictor.model_name,
            model_version=predictor.model_version,
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