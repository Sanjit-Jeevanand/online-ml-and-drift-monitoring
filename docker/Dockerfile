FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY scripts /app/scripts
COPY config /app/config
COPY artifacts /app/artifacts
COPY docker /app/docker

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]