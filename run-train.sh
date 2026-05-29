#!/usr/bin/env bash
# QLoRA DPO train on a HuntAI GPU node (16GB). Free the GPU from vLLM before running.
#
# Usage:
#   run-train.sh [OPTIONS] [BASE_MODEL] [OUTPUT_DIR] [-- train_dpo.py args...]
#
# Examples:
#   run-train.sh Qwen/Qwen2.5-1.5B-Instruct
#   run-train.sh Qwen/Qwen2.5-7B-Instruct checkpoints/my-run
#   run-train.sh -m Qwen/Qwen2.5-1.5B-Instruct -o checkpoints/test
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_ROOT="$(cd "$APP_ROOT/.." && pwd)"
ORCH_DPO="${ORCHESTRATOR_DPO_DIR:-$PLATFORM_ROOT/layer-orchestrator-v1/dpo-router}"
SCRIPTS="$APP_ROOT/scripts"
DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"

usage() {
  cat <<EOF
Usage: run-train.sh [OPTIONS] [BASE_MODEL] [OUTPUT_DIR] [-- train_dpo.py args...]

  BASE_MODEL   HuggingFace model id (default: ${DEFAULT_MODEL}, or env BASE_MODEL)
  OUTPUT_DIR   Checkpoint root (default: checkpoints/router-dpo-<slug>-<timestamp>)

Options:
  -m, --model ID         Base model (same as first positional)
  -o, --output-dir DIR   Output directory
  -h, --help             Show this help

Env: TRAIN_JSONL, VAL_JSONL, MAX_LENGTH, GRAD_ACCUM, NUM_TRAIN_EPOCHS, etc.

Wrappers: run-train-qwen25-7b.sh, run-train-qwen25-1.5b.sh
EOF
}

_slug_from_model() {
  local model="$1"
  local tail="${model##*/}"
  case "$tail" in
    *1.5B-Instruct* | *1.5b*) echo "qwen25-1.5b" ;;
    *7B-Instruct* | *7b*) echo "qwen25-7b" ;;
    *) echo "$tail" | tr '[:upper:]' '[:lower:]' | tr './' '-' ;;
  esac
}

default_output_dir() {
  local model="$1"
  echo "$APP_ROOT/checkpoints/router-dpo-$(_slug_from_model "$model")-$(date +%Y%m%d-%H%M)"
}

BASE_MODEL="${BASE_MODEL:-$DEFAULT_MODEL}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TRAIN_PY_EXTRA=()
POSITIONAL_MODEL=""
POSITIONAL_OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m | --model)
      BASE_MODEL="${2:?model id required after $1}"
      shift 2
      ;;
    -o | --output-dir)
      OUTPUT_DIR="${2:?directory required after $1}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      TRAIN_PY_EXTRA=("$@")
      break
      ;;
    -*)
      TRAIN_PY_EXTRA+=("$1")
      shift
      if [[ $# -gt 0 && "$1" != -* ]]; then
        TRAIN_PY_EXTRA+=("$1")
        shift
      fi
      ;;
    *)
      if [[ -z "$POSITIONAL_MODEL" ]]; then
        POSITIONAL_MODEL="$1"
        BASE_MODEL="$1"
      elif [[ -z "$POSITIONAL_OUTPUT" ]]; then
        POSITIONAL_OUTPUT="$1"
        OUTPUT_DIR="$1"
      else
        TRAIN_PY_EXTRA+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(default_output_dir "$BASE_MODEL")"
fi

if [[ -n "${TRAIN_JSONL:-}" ]]; then
  :
elif [[ -f "$APP_ROOT/data/train.jsonl" ]]; then
  TRAIN_JSONL="$APP_ROOT/data/train.jsonl"
  VAL_JSONL="${VAL_JSONL:-$APP_ROOT/data/val.jsonl}"
elif [[ -f "$ORCH_DPO/output/train.jsonl" ]]; then
  TRAIN_JSONL="$ORCH_DPO/output/train.jsonl"
  VAL_JSONL="${VAL_JSONL:-$ORCH_DPO/output/val.jsonl}"
else
  TRAIN_JSONL="$ORCH_DPO/output/train.jsonl"
  VAL_JSONL="${VAL_JSONL:-$ORCH_DPO/output/val.jsonl}"
fi

VENV="${VENV:-$APP_ROOT/.venv}"
PYTHON="${PYTHON:-}"

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Missing $TRAIN_JSONL" >&2
  echo "On a GPU-only host: bash fetch-dataset.sh" >&2
  echo "Or build locally: cd layer-orchestrator-v1 && bash dpo-router/run-build-dpo.sh" >&2
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

ARGS=(
  --train-jsonl "$TRAIN_JSONL"
  --val-jsonl "$VAL_JSONL"
  --base-model "$BASE_MODEL"
  --output-dir "$OUTPUT_DIR"
)
[[ -n "${MAX_LENGTH:-}" ]] && ARGS+=(--max-length "$MAX_LENGTH")
[[ -n "${NUM_TRAIN_EPOCHS:-}" ]] && ARGS+=(--num-train-epochs "$NUM_TRAIN_EPOCHS")
[[ -n "${PER_DEVICE_TRAIN_BATCH_SIZE:-}" ]] && ARGS+=(--per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE")
[[ -n "${GRAD_ACCUM:-}" ]] && ARGS+=(--gradient-accumulation-steps "$GRAD_ACCUM")

echo "train-jsonl=$TRAIN_JSONL" >&2
echo "val-jsonl=$VAL_JSONL" >&2
echo "base-model=$BASE_MODEL" >&2
echo "output-dir=$OUTPUT_DIR" >&2
[[ -n "${MAX_LENGTH:-}" ]] && echo "max-length=$MAX_LENGTH" >&2

exec "$PYTHON" "$SCRIPTS/train_dpo.py" "${ARGS[@]}" "${TRAIN_PY_EXTRA[@]}"
