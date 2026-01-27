from fastapi import FastAPI, HTTPException
from typing import Dict
import time
import json
import sys
from fastapi.responses import JSONResponse

from src.inference.predictor import Predictor
from src.inference.schemas import PredictRequest, PredictResponse


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

    model_name = "lightgbm"
    model_version = "v1.1.0"

    app.state.predictor = Predictor(
        model_name=model_name,
        model_version=model_version,
    )


# ============================================================
# Logging helper
# ============================================================

def log_event(payload: Dict) -> None:

    try:
        print(json.dumps(payload), flush=True)
    except Exception:
        print(
            json.dumps({
                "event": "logging_failed",
                "status": "error"
            }),
            file=sys.stderr,
            flush=True
        )


# ============================================================
# Routes
# ============================================================

@app.get("/health")
def health_check() -> Dict[str, str]:

    return {
        "status": "ok",
        "service": "inference",
        "version": "dev",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:

    request_start = time.perf_counter()

    predictor: Predictor = app.state.predictor

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

        log_event({
            "event": "inference_error",
            "status": status,
            **error_payload,
        })

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

        log_event({
            "event": "inference_error",
            "status": status,
            **error_payload,
        })

        return JSONResponse(
            status_code=500,
            content=error_payload,
        )

    finally:
        # -----------------------------
        # Final latency + logging
        # -----------------------------

        request_end = time.perf_counter()
        latency_ms = (request_end - request_start) * 1000.0

        log_event({
            "event": "inference_request",
            "model_name": predictor.model_name,
            "model_version": predictor.model_version,
            "status": status,
            "latency_ms": round(latency_ms, 3),
            "inference_ms": round(inference_ms, 3) if inference_ms is not None else None,
        })