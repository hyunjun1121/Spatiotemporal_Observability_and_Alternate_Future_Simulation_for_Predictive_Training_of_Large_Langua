#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python3}"
exec "$PYTHON_BIN" scripts/run_replicates.py "$@"
