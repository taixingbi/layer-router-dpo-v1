#!/bin/bash
# Deploy script run via SSM on EC2. Expects HF_TOKEN env var (optional).
# Run from /home/ubuntu/deploy after files are downloaded from S3.
set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/home/ubuntu/deploy}"
APP_DIR="${APP_DIR:-/home/ubuntu/layer-router-train-v1}"
DEPLOY_LOG="/tmp/router-train-deploy.log"

export DEBIAN_FRONTEND=noninteractive

exec > >(tee -a "$DEPLOY_LOG") 2>&1

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
# Drop cached JSONL so S3-synced data/output/ wins over stale data/{method}/.
rm -rf data/dpo data/sft

echo "=== Create fresh venv ==="
rm -rf .venv
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "=== Install GPU dependencies ==="
pip install -r "$DEPLOY_DIR/requirements-gpu.txt"
pip install -e . --no-deps

echo "=== Verify ML imports ==="
python -c "
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig
print('deps ok:', torch.__version__, 'cuda:', torch.cuda.is_available())
"

echo "=== Configure environment ==="
ENV_FILE="$APP_DIR/.env"
if [ ! -f "$ENV_FILE" ] && [ -f "$DEPLOY_DIR/.env.example" ]; then
  cp "$DEPLOY_DIR/.env.example" "$ENV_FILE"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  sed -i '/^HF_TOKEN=/d' "$ENV_FILE" 2>/dev/null || true
  printf 'HF_TOKEN=%s\n' "$HF_TOKEN" >> "$ENV_FILE"
fi
for _key in HF_REPO_ID HF_REPO_FEATURE HF_REPO_VERSION TRAIN_METHOD BASE_MODEL; do
  _val="${!_key:-}"
  if [ -n "$_val" ]; then
    sed -i "/^${_key}=/d" "$ENV_FILE" 2>/dev/null || true
    printf '%s=%s\n' "$_key" "$_val" >> "$ENV_FILE"
  fi
done
# Hub slug is derived from BASE_MODEL; drop stale overrides from older deploys.
sed -i '/^HF_REPO_MODEL=/d' "$ENV_FILE" 2>/dev/null || true

echo "=== Run router training (TRAIN_METHOD=${TRAIN_METHOD:-dpo}) ==="
METHOD="${TRAIN_METHOD:-dpo}"
BUILT_JSONL="$APP_DIR/data/output/$METHOD/train.jsonl"
CACHE_JSONL="$APP_DIR/data/$METHOD/train.jsonl"
if [ -f "$BUILT_JSONL" ]; then
  echo "dataset: $BUILT_JSONL ($(wc -l < "$BUILT_JSONL") lines, from S3 sync)"
elif [ -f "$CACHE_JSONL" ]; then
  echo "dataset: $CACHE_JSONL (cached)"
else
  echo "WARN: $BUILT_JSONL missing after S3 sync; app.train will fetch from GitHub (data/output/$METHOD/)"
fi
cd "$APP_DIR"
PYTHONUNBUFFERED=1 python -u -m app.train.main

echo "=== Done ==="
ls -la checkpoints/ 2>/dev/null || true
