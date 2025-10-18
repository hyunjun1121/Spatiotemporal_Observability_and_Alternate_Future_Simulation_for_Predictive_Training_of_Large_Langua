# ICLR Reproducibility Checklist (RC2)

## Code & Artifacts
- [x] Training/Evaluation code (`src/**`) 공개
- [x] Orchestration 스크립트 (`scripts/run_replicates.sh`, `Makefile lock`) 제공
- [x] Result Lock 산출물 (`lock.json`, `paper/tables/*.tsv`, `paper/RESULT_LOCK.md`) 포함

## Data
- [x] Real data 프로필 명시: 미집계
- [x] 환경 설정 (`assets/experiments/*.yaml`, `environment.yml`) 제공

## Hardware & Compute
- [x] run_manifest.json에 device/CUDA 기록
- [x] branch orchestration 로그/요약(`runs/**/summary/summary.{json,md}`) 제공

## Seeds & Determinism
- [x] 고정 seed(1337/2337/3337) replicate 수행
- [x] seed_manifest.json에 random seed 기록

## Logging & Checkpoints
- [x] train_log.jsonl 및 checkpoints(`runs/**/branches/`) 저장
- [x] summary.json/plots_index.json으로 metrics 추적

## Statistical Methods
- [x] paired t-test, Cliff's delta, bootstrap CI 적용 (src/eval/stats.py)
- [x] Result Lock 기준(instability, wasted FLOPs, val pplx) 명시

## Limitations & Ethics
- [x] Limitations 섹션에서 proxy miscalibration/compute overhead/data bias 다룸
- [x] Ethics Statement 섹션 초안 예약

Generated at 2025-10-18T00:55:50.157859Z
