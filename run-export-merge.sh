#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$APP_ROOT/.venv}"
PYTHON="${PYTHON:-${VENV}/bin/python}"

exec "$PYTHON" "$APP_ROOT/scripts/export_merge.py" "$@"
