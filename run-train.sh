#!/usr/bin/env bash
# QLoRA DPO train on a HuntAI GPU node (16GB). Free the GPU from vLLM before running.
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_ROOT="$(cd "$APP_ROOT/.." && pwd)"
ORCH_DPO="${ORCHESTRATOR_DPO_DIR:-$PLATFORM_ROOT/layer-orchestrator-v1/dpo-router}"
SCRIPTS="$APP_ROOT/scripts"

TRAIN_JSONL="${TRAIN_JSONL:-$ORCH_DPO/output/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-$ORCH_DPO/output/val.jsonl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$APP_ROOT/checkpoints/router-dpo-$(date +%Y%m%d-%H%M)}"
VENV="${VENV:-$APP_ROOT/.venv}"
PYTHON="${PYTHON:-}"

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Missing $TRAIN_JSONL" >&2
  echo "Build dataset: cd layer-orchestrator-v1 && bash dpo-router/run-build-dpo.sh" >&2
  exit 1
fi

if [[ -z "$PYTHON" ]]; then
  if [[ -x "$VENV/bin/python" ]]; then
    PYTHON="$VENV/bin/python"
  else
    PYTHON=python3
  fi
fi

if ! "$PYTHON" -c "import torch, trl, peft" 2>/dev/null; then
  echo "Training deps missing. Create venv and install:" >&2
  echo "  python3.11 -m venv $VENV && $VENV/bin/pip install -r $APP_ROOT/requirements.txt" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "train-jsonl=$TRAIN_JSONL" >&2
echo "val-jsonl=$VAL_JSONL" >&2
echo "base-model=$BASE_MODEL" >&2
echo "output-dir=$OUTPUT_DIR" >&2

exec "$PYTHON" "$SCRIPTS/train_dpo.py" \
  --train-jsonl "$TRAIN_JSONL" \
  --val-jsonl "$VAL_JSONL" \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
