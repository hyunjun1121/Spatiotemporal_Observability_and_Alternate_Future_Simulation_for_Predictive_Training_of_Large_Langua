# Spatiotemporal Observability (RC2)

## Quickstart

```bash
make smoke            # synthetic sanity run
make test             # pytest + unit checks
make ablate           # mode A/B/C/D ablations
make replay CP_PATH=... DECISION=...  # branch replay utility
```

## Result Lock Workflow

```bash
make lock                   # runs wikitext_rc1 + c4small_rc1 pipelines
python -m scripts.gen_paper_assets  # 업데이트된 paper/iclr_draft.md + checklist
```

- 산출물: `paper/tables/results.tsv`, `paper/tables/stats.tsv`, `paper/RESULT_LOCK.md`, `lock.json`.
- bar plot 포함 시각 자료: `paper/figs/bar_*.png`.
- 빠른 Result Lock: `make real-wt103-fastlock`, `make real-c4small-fastlock`(동시 실행 2, budget_stop, retries 2).

### Scheduler Options

- `--max_concurrent <N>`: 동시에 실행할 조합 개수 상한(GPU 공유 시 oversubscribe 주의).
- `--devices "0,1,2,3"`: `CUDA_VISIBLE_DEVICES` 마스킹.
- `--world_size_auto`: device 개수 기준으로 DDP world size 자동 결정.
- `--budget_stop`: fixedlr vs method=C가 Result Lock 조건 충족 시 남은 조합 자동 skip.
- `--retries <K>`: 실패 시 재시도 횟수 지정(기본 0).
- `--resume`: seed별 `.done`/결과 기록이 있으면 skip 하여 이어서 실행.

## Real Data Replication

```bash
make real-wt103-lock      # wikitext_rc1, baselines × methods × 3 seeds
make real-c4small-lock    # c4small_rc1 동일 구성
make real-wt103-fastlock  # 조기락 활성화, max concurrency=2
make real-c4small-fastlock
```

- Hugging Face streaming uses `Salesforce/wikitext` (config `wikitext-103-v1`). Ensure `pip install datasets` and `hf auth login` or preload cache (`HF_HOME`) before running real-data pipelines.

- Set `NCCL_P2P_DISABLE=1` on GPU nodes (e.g., `export NCCL_P2P_DISABLE=1`).

- Launch the tmux automation via:
  ```bash
  tmux new-session -d -s realrun "bash -lc '~/Spatiotemporal_Observability_and_Alternate_Future_Simulation_for_Predictive_Training_of_Large_Langua/run_real_experiments.sh'"
  ```
  Attach with `tmux attach -t realrun` and logs go to `logs/*.log`.

- Script: `run_real_experiments.sh` (wikitext fastlock → c4 fastlock → `make paper`).

- 단일 조합 검증은 `scripts/run_replicates.sh --real_data ... --baseline ... --method ... --config ...` 활용.

## Paper & Documentation

```bash
python -m scripts.gen_paper_assets
```

- 생성 파일: `paper/iclr_draft.md`, `paper/ICLR_CHECKLIST.md`, 최신 Ablation bullet, Limitations 자동 반영.

## Docker Build

```bash
docker build -t stobs:rc2 --build-arg EXTRAS=1 .  # pandas/pyarrow 포함
```

## Encoding Check

```bash
python scripts/check_encoding.py paper
```
