#!/usr/bin/env bash
set -e

echo "Starting inference service..."

exec uvicorn src.inference.service:app \
    --host 0.0.0.0 \
    --port 8000