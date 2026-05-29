#!/bin/bash
# Deploy script run via SSM on EC2. Expects HF_TOKEN env var (optional).
# Run from /home/ubuntu/deploy after files are downloaded from S3.
set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/home/ubuntu/deploy}"
APP_DIR="${APP_DIR:-/home/ubuntu/layer-router-dpo-v1}"

echo "=== GPU check ==="
nvidia-smi || { echo "ERROR: nvidia-smi failed"; exit 1; }

echo "=== Install Python tooling ==="
PY_VER="$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)"
sudo apt-get update -y
sudo apt-get install -y "python${PY_VER}-venv" python3-pip

echo "=== Prepare app directory ==="
cd "$APP_DIR" || { echo "ERROR: $APP_DIR not found"; exit 1; }
mkdir -p data checkpoints

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

echo "=== Configure environment ==="
ENV_FILE="$APP_DIR/.env"
if [ ! -f "$ENV_FILE" ] && [ -f "$DEPLOY_DIR/.env.example" ]; then
  cp "$DEPLOY_DIR/.env.example" "$ENV_FILE"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  sed -i '/^HF_TOKEN=/d' "$ENV_FILE" 2>/dev/null || true
  printf 'HF_TOKEN=%s\n' "$HF_TOKEN" >> "$ENV_FILE"
fi

echo "=== Run DPO training ==="
cd "$APP_DIR"
python -m app.main

echo "=== Done ==="
ls -la checkpoints/ 2>/dev/null || true
