# Phase 1 Report

## Setup

- Single node, 1×GPU smoke(default synthetic)
- Torch version, Python 3.11, optional extras(pyarrow/fastparquet)
- Launch
  - Single GPU: `python -m src.main --mode baseline --dataset synthetic --steps 400 --batch_size 8 --seq_len 128`
  - 2 GPU DDP: `torchrun --nproc_per_node=2 -m src.main --mode baseline --dataset synthetic --steps 800`

## Configs

- Assets: `assets/state_schema_v1.yaml`, `assets/uncertainty_config.yaml`, `assets/hparam_space.yaml`, `assets/branch_policy.yaml`, `assets/noise_config.yaml`, `assets/proxy_config.yaml`
- Logging: JSONL at `runs/train_log.jsonl`, Parquet at `runs/log_db.parquet`

## Repro Steps

- `make smoke` → synthetic smoke run. 결과는 `runs/<experiment>/<run_id>/summary/summary.md`, `summary/plots/*.png`에서 확인.
- `make test` → pytest suite 실행(시드/체크포인트/uncertainty/proxy/bandit).
- `make replay CP_PATH=... DECISION=...` → deterministic replay JSON + Δ지표 요약 확인.
- `make ablate` → 모드 A/B/C/D 자동 실행(`runs/ablations/<stamp>/run_<mode>/summary/summary.md`, `summary.tsv`, `index.md`).

## Results

_(실험별 summary.md/plots를 첨부)_

## Ablations

- 자동 스크립트: `make ablate`
- 산출물: `runs/ablations/<stamp>/run_<mode>/summary/summary.md`, `runs/ablations/<stamp>/summary.tsv`, `runs/ablations/<stamp>/index.md`

## Failures

- HF datasets unavailable → synthetic fallback
- Parquet unavailable → JSONL fallback

## Lessons

- Uncertainty-triggered branching + Proxy rollout이 smoke 수준에서 안정
- Entropy/forecaster 업데이트는 subsample(50 step)로 오버헤드 제어

## Deterministic Replay 결과 템플릿

- 명령: `python -m src.runner.replay --cp_path runs/<exp>/<run_id>/branches/<t>/cp_<t>.ptz --decision runs/<exp>/<run_id>/branches/<t>/decision.json`
- 허용 오차: `max|Δloss|`, `max|Δgrad_norm|` ≤ 1e-5 → `tolerance_met: true`
- 요약: `runs/<exp>/<run_id>/summary/summary.md` 부록에 추가

## Proxy Metrics

- Offline dataset builder(`src/proxy/dataset.py`) → rolling window 생성
- ProxyMLP(`src/proxy/train.py`) → `mae_loss`, `mae_risk`, `mae_recovery`, `ece`
