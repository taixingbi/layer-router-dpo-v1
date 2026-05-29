#!/bin/bash
# Deploy script run via SSM on EC2. Expects HF_TOKEN env var (optional).
# Run from /home/ubuntu/deploy after files are downloaded from S3.
set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/home/ubuntu/deploy}"
APP_DIR="${APP_DIR:-/home/ubuntu/layer-router-dpo-v1}"
HF_REPO_ID="${HF_REPO_ID:-taixingbi/layer-router-dpo-v1}"

export DEBIAN_FRONTEND=noninteractive

_find_python() {
  local cmd ver
  for cmd in python3.12 python3.11 python3; do
    command -v "$cmd" >/dev/null 2>&1 || continue
    ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if "$cmd" -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)'; then
      echo "$cmd"
      return 0
    fi
    echo "skip $cmd ($ver, need >=3.11)" >&2
  done
  return 1
}

echo "=== GPU check ==="
nvidia-smi || { echo "ERROR: nvidia-smi failed"; exit 1; }

echo "=== Install Python 3.11+ ==="
if ! PYTHON="$(_find_python)"; then
  sudo apt-get update -y
  if sudo apt-get install -y python3.11 python3.11-venv python3.11-dev; then
    PYTHON=python3.11
  else
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    PYTHON=python3.11
  fi
fi
echo "Using $PYTHON ($($PYTHON --version))"

echo "=== Prepare app directory ==="
cd "$APP_DIR" || { echo "ERROR: $APP_DIR not found"; exit 1; }
mkdir -p data checkpoints

if [ -d .venv ] && ! .venv/bin/python -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  echo "Removing .venv (Python <3.11)"
  rm -rf .venv
fi
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
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
sed -i '/^HF_REPO_ID=/d' "$ENV_FILE" 2>/dev/null || true
printf 'HF_REPO_ID=%s\n' "$HF_REPO_ID" >> "$ENV_FILE"

echo "=== Run DPO training ==="
cd "$APP_DIR"
python -m app.main

echo "=== Done ==="
ls -la checkpoints/ 2>/dev/null || true
