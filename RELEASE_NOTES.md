# Release Notes — RC2

## 하이라이트
- scripts/run_replicates.sh + Makefile lock 타깃으로 real-data 3 seeds 자동 실행 및 집계.
- src/eval.aggregate 개선: Result Lock 기준(statistics, CI bar plots, lock.json) 자동 산출.
- scripts/gen_paper_assets.py로 Abstract/Results/Ablations/Limitations 자동 채움 및 ICLR checklist 생성.
- Dockerfile ARG EXTRAS 옵션으로 parquet/pandas 의존성 온디맨드 설치 지원.

## Result Lock 요약
- Result Lock 평가는 make lock 실행 후 paper/RESULT_LOCK.md와 lock.json에 기록됩니다.
- method C vs fixedlr(Primary baseline) 기준: instability_events_per_100k, wasted_flops_est_total, val_pplx 비교 통계가 자동 반영됩니다.

## 재현 절차
1. make lock
2. python -m scripts.gen_paper_assets
3. (선택) docker build -t stobs:rc2 --build-arg EXTRAS=1 .

## 산출물 위치
- 집계 테이블: paper/tables/results.tsv, paper/tables/stats.tsv
- Result Lock 문서: paper/RESULT_LOCK.md, lock.json
- Paper draft & checklist: paper/iclr_draft.md, paper/ICLR_CHECKLIST.md
- 그림: paper/figs/bar_*.png + run summary plots

## 알려진 이슈
- 실제 real-data 다운로드/캐시 경로는 repo에 포함되지 않음 (사전 준비 필요).
- long-running replicates는 GPU 리소스를 요구하며, make lock 전 sandbox 용량을 확인해야 함.
- Result Lock 수치는 파이프라인 실행 이전에는 갱신되지 않습니다.

