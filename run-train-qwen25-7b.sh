#!/usr/bin/env bash
# QLoRA DPO — Qwen2.5-7B (matches prod LLM_MODEL / vLLM base).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
exec bash "$ROOT/run-train.sh" "Qwen/Qwen2.5-7B-Instruct" "$@"
