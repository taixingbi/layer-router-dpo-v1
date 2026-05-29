#!/usr/bin/env bash
# Download committed DPO JSONL from layer-orchestrator-v1 (GPU node may not have that repo).
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$APP_ROOT/data}"
REF="${ORCHESTRATOR_DPO_REF:-main}"
BASE_URL="https://raw.githubusercontent.com/taixingbi/layer-orchestrator-v1/${REF}/dpo-router/output"

mkdir -p "$DATA_DIR"
for name in train.jsonl val.jsonl build-stats.json; do
  echo "fetch $name" >&2
  curl -fsSL "$BASE_URL/$name" -o "$DATA_DIR/$name"
done
echo "dataset -> $DATA_DIR" >&2
