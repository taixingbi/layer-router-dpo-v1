#!/usr/bin/env bash
# QLoRA DPO — Qwen2.5-1.5B (easiest on 16GB / RTX 3090).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
exec bash "$ROOT/run-train.sh" "Qwen/Qwen2.5-1.5B-Instruct" "$@"
