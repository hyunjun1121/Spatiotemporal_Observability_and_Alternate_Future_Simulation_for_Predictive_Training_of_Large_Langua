#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 && ( -z "${CP_PATH:-}" || -z "${DECISION:-}" ) ]]; then
  cat <<'USAGE' >&2
Usage:
  make replay CP_PATH=<checkpoint> DECISION=<decision.json>
  or
  scripts/run_replay.sh --cp_path <checkpoint> --decision <decision.json> [--steps N]
USAGE
  exit 1
fi

if [[ $# -ge 2 ]]; then
  python -m src.runner.replay "$@"
else
  python -m src.runner.replay --cp_path "${CP_PATH}" --decision "${DECISION}"
fi
