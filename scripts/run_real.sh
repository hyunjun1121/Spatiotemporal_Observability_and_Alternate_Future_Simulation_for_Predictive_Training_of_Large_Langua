#!/usr/bin/env bash
set -euo pipefail

while [[ $# -gt 0 ]]; do
  case "$1" in
    --real_data) REAL_DATA="$2"; shift 2 ;;
    --replicates) REPS="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

REAL_DATA=${REAL_DATA:-wikitext}
REPS=${REPS:-3}
BASELINE=${BASELINE:-none}
METHOD=${METHOD:-C}
CONFIG=${CONFIG:-assets/experiments/wikitext_rc1.yaml}

SEQ_LEN=1024
STEPS=50000
BATCH=16
if [[ -f "${CONFIG}" ]]; then
  if python -c "import yaml" 2>/dev/null; then
    while IFS='=' read -r key val; do
      case "$key" in
        seq_len) SEQ_LEN="$val" ;;
        steps) STEPS="$val" ;;
        batch_size) BATCH="$val" ;;
      esac
    done < <(python - <<'PY'
import yaml, sys
cfg = yaml.safe_load(open(sys.argv[1]))
for k in ("seq_len", "steps", "batch_size"):
    if k in cfg:
        print(f"{k}={cfg[k]}")
PY
"${CONFIG}")
  fi
fi

RUN_PREFIX="runs/${REAL_DATA}_${BASELINE}_${METHOD}"

python -m src.main --mode baseline --dataset hf --real_data "${REAL_DATA}" --replicates "${REPS}" \
  --baseline "${BASELINE}" --method "${METHOD}" --experiment_name "${RUN_PREFIX}" --steps "${STEPS}" --seq_len "${SEQ_LEN}" --batch_size "${BATCH}"

python -m scripts.gen_paper_assets --runs_root runs --paper_dir paper

echo "Summary tables:"
ls paper/tables
echo "Run summaries:"
python - <<'PY'
from pathlib import Path
for path in Path("runs").rglob("summary.md"):
    print(path)
PY
