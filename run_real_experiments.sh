#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

module load gcc/13.2.0
module load cuda/12.4.1
source "$PROJECT_ROOT/.venv/bin/activate"
export NCCL_P2P_DISABLE=1

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

run_job() {
  local name="$1"
  shift
  echo "[$(date)] Starting $name" | tee -a "$LOG_DIR/${name}.log"
  "$@" >> "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  echo "[$(date)] Finished $name (exit $status)" | tee -a "$LOG_DIR/${name}.log"
  return $status
}

run_job wikitext_fastlock env CUDA_VISIBLE_DEVICES=0 bash scripts/run_replicates.sh \
  --real_data wikitext \
  --baseline fixedlr,hypergrad,zclip,spamlite,pbtlite \
  --method A,B,C,D \
  --config assets/experiments/wikitext_rc1.yaml \
  --max_concurrent 1 \
  --budget_stop \
  --retries 2

run_job c4_fastlock env CUDA_VISIBLE_DEVICES=0 bash scripts/run_replicates.sh \
  --real_data c4 \
  --baseline fixedlr,hypergrad,zclip,spamlite,pbtlite \
  --method A,B,C,D \
  --config assets/experiments/c4small_rc1.yaml \
  --max_concurrent 1 \
  --budget_stop \
  --retries 2

run_job make_paper make paper

echo "All experiments complete at $(date)" | tee -a "$LOG_DIR/summary.log"
