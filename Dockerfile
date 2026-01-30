FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY scripts /app/scripts
COPY config /app/config
COPY artifacts /app/artifacts

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8080

CMD ["sh", "-c", "echo '=== CONTAINER STARTED ===' && python -c \"import src.inference.service; print('=== IMPORT OK ===')\" && uvicorn src.inference.service:app --host 0.0.0.0 --port ${PORT:-8000}"]