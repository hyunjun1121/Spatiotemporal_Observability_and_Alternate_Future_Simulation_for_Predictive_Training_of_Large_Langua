#!/usr/bin/env bash
set -eu
if ! set -o pipefail >/dev/null 2>&1; then
  :
fi

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN=${PYTHON}
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
  else
    echo "Python interpreter not found" >&2
    exit 1
  fi
fi
if command -v cygpath >/dev/null 2>&1; then
  case "${PYTHON_BIN}" in
    [A-Za-z]:\\*|[A-Za-z]:/*)
      PYTHON_BIN=$(cygpath -u "${PYTHON_BIN}")
      ;;
  esac
fi

ALL_BASELINES=(fixedlr hypergrad zclip spamlite pbtlite)
ALL_METHODS=(A B C D)

show_help() {
  cat <<'EOF'
Usage: scripts/run_replicates.sh [options]

Options:
  --real_data {wikitext,c4,off}    Real-data profile (default: wikitext)
  --baseline LIST                  Baseline name or comma-separated list (default: fixedlr)
  --method LIST                    Method variant or comma-separated list (default: C)
  --config PATH                    Experiment config YAML (default: inferred from real_data)
  --runs_root PATH                 Root directory for runs (default: runs)
  --paper_dir PATH                 Output directory for paper assets (default: paper)
  --groupby COLS                   Aggregation grouping columns (default: experiment,baseline,method)
  --dry-run                        Create directory structure without executing training
  -h, --help                       Show this help message

This script launches 3 replicate runs (seeds 1337/2337/3337) for each baseline/method
combination, materialising run directories under runs/<experiment>/<baseline>/<method>/seed<seed>/...
After completion, src.eval.aggregate is invoked to update paper tables.
EOF
}

REAL_DATA="wikitext"
BASELINE_ARG=""
METHOD_ARG=""
CONFIG=""
RUNS_ROOT="runs"
PAPER_DIR="paper"
GROUPBY="experiment,baseline,method"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --real_data)
      REAL_DATA="$2"
      shift 2
      ;;
    --baseline)
      BASELINE_ARG="$2"
      shift 2
      ;;
    --method)
      METHOD_ARG="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --runs_root)
      RUNS_ROOT="$2"
      shift 2
      ;;
    --paper_dir)
      PAPER_DIR="$2"
      shift 2
      ;;
    --groupby)
      GROUPBY="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${BASELINE_ARG}" ]]; then
  BASELINE_SET=("fixedlr")
elif [[ "${BASELINE_ARG}" == "all" ]]; then
  BASELINE_SET=("${ALL_BASELINES[@]}")
else
  IFS=',' read -r -a BASELINE_SET <<< "${BASELINE_ARG}"
fi

if [[ -z "${METHOD_ARG}" ]]; then
  METHOD_SET=("C")
elif [[ "${METHOD_ARG}" == "all" ]]; then
  METHOD_SET=("${ALL_METHODS[@]}")
else
  IFS=',' read -r -a METHOD_SET <<< "${METHOD_ARG}"
fi

real_data_lower=$(echo "${REAL_DATA}" | tr '[:upper:]' '[:lower:]')
case "${real_data_lower}" in
  wikitext)
    DEFAULT_CONFIG="assets/experiments/wikitext_rc1.yaml"
    ;;
  c4)
    DEFAULT_CONFIG="assets/experiments/c4small_rc1.yaml"
    ;;
  off|synthetic)
    DEFAULT_CONFIG="assets/experiments/synthetic_quick.yaml"
    ;;
  *)
    echo "Unsupported real_data value: ${REAL_DATA}" >&2
    exit 1
    ;;
esac

CONFIG=${CONFIG:-${DEFAULT_CONFIG}}
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config file not found: ${CONFIG}" >&2
  exit 1
fi

read_config() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception as exc:
    raise SystemExit(f"PyYAML required to parse config: {exc}")

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

for key in ("dataset", "real_data", "hf_name", "seq_len", "batch_size", "steps", "replicates", "experiment", "experiment_name", "cooldown", "epsilon_ucb"):
    if key in data:
        print(f"{key}={data[key]}")
PY
}

CONFIG_DATA=()
while IFS= read -r line; do
  CONFIG_DATA+=("$line")
done < <(read_config "${CONFIG}")

DATASET="hf"
CFG_REAL_DATA="${REAL_DATA}"
HF_NAME="wikitext-103"
SEQ_LEN=1024
BATCH_SIZE=16
STEPS=50000
REPLICATES=3
EXPERIMENT_NAME="$(basename "${CONFIG%.*}")"
COOLDOWN=300
EPSILON_UCB=0.1

for entry in "${CONFIG_DATA[@]}"; do
  key="${entry%%=*}"
  value="${entry#*=}"
  case "${key}" in
    dataset) DATASET="${value}" ;;
    real_data) CFG_REAL_DATA="${value}" ;;
    hf_name) HF_NAME="${value}" ;;
    seq_len) SEQ_LEN="${value}" ;;
    batch_size) BATCH_SIZE="${value}" ;;
    steps) STEPS="${value}" ;;
    replicates) REPLICATES="${value}" ;;
    experiment_name) EXPERIMENT_NAME="${value}" ;;
    experiment) EXPERIMENT_NAME="${value}" ;;
    cooldown) COOLDOWN="${value}" ;;
    epsilon_ucb) EPSILON_UCB="${value}" ;;
  esac
done

# Allow config to override CLI-provided real_data
REAL_DATA="${CFG_REAL_DATA:-${REAL_DATA}}"

readarray -t SEEDS < <(printf "%s\n" 1337 2337 3337)

mkdir -p "${RUNS_ROOT}"
mkdir -p "${PAPER_DIR}/tables"
mkdir -p "${PAPER_DIR}/figs"

run_training() {
  local baseline="$1"
  local method="$2"
  local seed="$3"
  local base_dir="${RUNS_ROOT}/${EXPERIMENT_NAME}/${baseline}/${method}/seed${seed}"
  mkdir -p "${base_dir}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    # Touch manifest placeholder for tests
    touch "${base_dir}/.dry_run"
    return
  fi

  "$PYTHON_BIN" -m src.main \
    --mode baseline \
    --dataset "${DATASET}" \
    --real_data "${REAL_DATA}" \
    --hf_name "${HF_NAME}" \
    --seq_len "${SEQ_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --steps "${STEPS}" \
    --seed "${seed}" \
    --replicates 1 \
    --baseline "${baseline}" \
    --method "${method}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --run_dir "${base_dir}" \
    --cooldown "${COOLDOWN}" \
    --epsilon_ucb "${EPSILON_UCB}" \
    --config "${CONFIG}"
}

for baseline in "${BASELINE_SET[@]}"; do
  case " ${ALL_BASELINES[*]} " in
    *" ${baseline} "* ) ;;
    *)
      echo "Unknown baseline: ${baseline}" >&2
      exit 1
      ;;
  esac
  for method in "${METHOD_SET[@]}"; do
    case " ${ALL_METHODS[*]} " in
      *" ${method} "* ) ;;
      *)
        echo "Unknown method: ${method}" >&2
        exit 1
        ;;
    esac
    for seed in "${SEEDS[@]}"; do
      run_training "${baseline}" "${method}" "${seed}"
    done
  done
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  "$PYTHON_BIN" -m src.eval.aggregate \
    --runs_root "${RUNS_ROOT}" \
    --paper_dir "${PAPER_DIR}" \
    --groupby "${GROUPBY}"
fi
