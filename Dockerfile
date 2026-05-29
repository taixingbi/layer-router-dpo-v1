# GPU batch image for QLoRA DPO training (requires `docker run --gpus all`).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ARG APP_VERSION=dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_VERSION=${APP_VERSION} \
    HF_HOME=/cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN mkdir -p /app/data /app/checkpoints /cache/huggingface \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app /cache

USER appuser

VOLUME ["/app/data", "/app/checkpoints", "/cache/huggingface"]

ENTRYPOINT ["python", "-m", "app.main"]
