#!/usr/bin/env bash
set -euo pipefail

STAMP=$(date +"%Y%m%d_%H%M%S")
BASE_RUN_DIR="runs/ablations/${STAMP}"
mkdir -p "${BASE_RUN_DIR}"

MODES=(A B C D)

for MODE in "${MODES[@]}"; do
  echo "[ablation] running mode ${MODE}"
  RUN_PATH="${BASE_RUN_DIR}/run_${MODE}"
  python -m src.main --mode baseline --dataset synthetic --steps 400 --batch_size 8 --seq_len 128 \
    --ablation_mode "${MODE}" --experiment_name "ablation_${MODE}" --run_dir "${RUN_PATH}"
  python -m src.eval.aggregate --run_dir "${RUN_PATH}"
done

python -m src.runner.ablation --root "${BASE_RUN_DIR}" --out "${BASE_RUN_DIR}/summary"

INDEX_FILE="${BASE_RUN_DIR}/index.md"
echo "# Ablation runs (${STAMP})" > "${INDEX_FILE}"
echo "" >> "${INDEX_FILE}"
for MODE in "${MODES[@]}"; do
  echo "- ${MODE}: [summary](run_${MODE}/summary/summary.md)" >> "${INDEX_FILE}"
done

python - <<'PY'
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
rows = []
for mode in ["A", "B", "C", "D"]:
    summary_path = base / f"run_{mode}" / "summary" / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["mode"] = mode
        rows.append(data)

tsv_path = base / "summary.tsv"
if rows:
    headers = ["mode", "steps", "instability_events_per_100k", "branch_invoke_rate", "accept_rate", "proxy_real_gap_mean", "wasted_flops_est_total"]
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")
PY" "${BASE_RUN_DIR}"

echo "Ablation results stored in ${BASE_RUN_DIR}"
