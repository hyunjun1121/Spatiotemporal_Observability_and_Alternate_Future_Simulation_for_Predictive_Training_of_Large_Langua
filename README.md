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

## Real Data Replication

```bash
make real-wt103-lock      # wikitext_rc1, baselines × methods × 3 seeds
make real-c4small-lock    # c4small_rc1 동일 구성
```

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
