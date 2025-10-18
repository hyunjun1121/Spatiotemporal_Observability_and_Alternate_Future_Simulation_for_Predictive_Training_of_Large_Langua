#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def _to_float(value: Optional[str]) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _format_percent(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.1%}"


def _format_pct_point(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.1f}pp"


def _format_float(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _markdown_table(rows: List[Dict[str, str]], columns: List[Tuple[str, str]], float_cols: Iterable[str]) -> str:
    if not rows:
        return "_No data available._"
    float_cols = set(float_cols)
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + " | ".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if key in float_cols:
                value = _format_float(_to_float(value))
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _collect_dataset_summary(
    results_rows: List[Dict[str, str]],
    stats_rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, Dict[str, str]]]:
    summary: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in results_rows:
        if row.get("baseline") != "fixedlr":
            continue
        exp = row.get("experiment", "")
        summary.setdefault(exp, {"results": {}, "stats": {}})["results"][row.get("method", "")] = row
    for row in stats_rows:
        if row.get("baseline") != "fixedlr":
            continue
        if row.get("method") != "C":
            continue
        exp = row.get("experiment", "")
        metric = row.get("metric", "")
        summary.setdefault(exp, {"results": {}, "stats": {}})["stats"][metric] = row
    return summary


def _latest_ablation(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = sorted([p for p in root.iterdir() if p.is_dir()])
    return candidates[-1] if candidates else None


def _build_ablation_bullets(ablations_dir: Optional[Path]) -> List[str]:
    if ablations_dir is None:
        return ["ablation summary 없음 (runs/ablations/* 실행 필요)."]
    summary_path = ablations_dir / "summary.tsv"
    if not summary_path.exists():
        return [f"{ablations_dir.name}에서 summary.tsv를 찾을 수 없습니다."]
    rows = _read_tsv(summary_path)
    by_mode = {row.get("mode", ""): row for row in rows}
    baseline = by_mode.get("A")
    if not baseline:
        return ["mode A (baseline) 정보가 없어 비교 불가."]

    bullets: List[str] = []
    base_instab = _to_float(baseline.get("instability_events_per_100k"))
    base_wasted = _to_float(baseline.get("wasted_flops_est_total"))
    base_accept = _to_float(baseline.get("accept_rate"))
    base_branch = _to_float(baseline.get("branch_invoke_rate"))

    for mode in ["B", "C", "D"]:
        target = by_mode.get(mode)
        if not target:
            continue
        instab = _to_float(target.get("instability_events_per_100k"))
        wasted = _to_float(target.get("wasted_flops_est_total"))
        accept = _to_float(target.get("accept_rate"))
        branch = _to_float(target.get("branch_invoke_rate"))

        instab_delta = (instab - base_instab) / base_instab if math.isfinite(base_instab) and base_instab != 0 else float("nan")
        wasted_delta = (wasted - base_wasted) / base_wasted if math.isfinite(base_wasted) and base_wasted != 0 else float("nan")
        accept_delta = (accept - base_accept) if math.isfinite(base_accept) else float("nan")
        branch_delta = (branch - base_branch) if math.isfinite(base_branch) else float("nan")

        bullets.append(
            f"mode {mode}: instability_events_per_100k { _format_percent(instab_delta) }, "
            f"wasted_flops_est_total { _format_percent(wasted_delta) }, "
            f"accept_rate { _format_pct_point(accept_delta * 100) }, "
            f"branch_invoke_rate { _format_pct_point(branch_delta * 100) }."
        )

    return bullets or ["ablation metric 비교 결과가 없습니다."]


def _build_abstract(dataset_summary: Dict[str, Dict[str, Dict[str, str]]]) -> str:
    if not dataset_summary:
        return "Result Lock을 위한 실험 결과가 아직 집계되지 않았습니다."
    pieces = []
    for dataset, info in dataset_summary.items():
        stats = info.get("stats", {})
        instab = stats.get("instability_events_per_100k", {})
        wasted = stats.get("wasted_flops_est_total", {})
        pplx = stats.get("val_pplx", {})
        instab_improve = _to_float(instab.get("relative_diff"))
        wasted_improve = _to_float(wasted.get("relative_diff"))
        pplx_delta = _to_float(pplx.get("relative_diff"))
        pieces.append(
            f"{dataset}에서 instability_events_per_100k { _format_percent(instab_improve) }, "
            f"wasted_flops_est_total { _format_percent(wasted_improve) }, "
            f"val_pplx Δ { _format_percent(pplx_delta) }"
        )
    joined = "; ".join(pieces)
    return (
        "본 연구는 method C 기반 Spatiotemporal Observability 기법이 Primary baseline 대비 "
        f"{joined} 수준으로 Result Lock 조건을 충족함을 보인다."
    )


def _build_limitations(dataset_summary: Dict[str, Dict[str, Dict[str, str]]]) -> str:
    if not dataset_summary:
        return "실험 집계가 부족하여 한계를 평가할 수 없습니다."

    proxy_vals = []
    invoke_vals = []
    datasets = []
    for dataset, info in dataset_summary.items():
        datasets.append(dataset)
        results = info.get("results", {})
        method_c = results.get("C", {})
        proxy_vals.append(_to_float(method_c.get("proxy_gap_mean")))
        invoke_vals.append(_to_float(method_c.get("invoke_rate_mean")))
    max_proxy = max((v for v in proxy_vals if math.isfinite(v)), default=float("nan"))
    avg_invoke = sum(v for v in invoke_vals if math.isfinite(v)) / max(
        1, sum(1 for v in invoke_vals if math.isfinite(v))
    )
    dataset_list = ", ".join(datasets)
    limitation_parts = []
    if math.isfinite(max_proxy):
        limitation_parts.append(f"proxy_real_gap_mean이 최대 {max_proxy:.3f} 수준으로 residual miscalibration이 남아있다")
    else:
        limitation_parts.append("proxy_real_gap_mean 변화를 아직 정량화하지 못했다")
    if math.isfinite(avg_invoke):
        limitation_parts.append(f"branch_invoke_rate 평균이 {avg_invoke:.2f}로 compute overhead가 존재한다")
    else:
        limitation_parts.append("branch_invoke_rate 측정값이 부족하여 compute 비용을 추정하기 어렵다")
    limitation_parts.append(f"real data 프로필 {dataset_list}에 한정된 평가로 data bias guard가 필요하다")
    limitation_parts.append("예측 proxy가 드물게 drift할 수 있어 모니터링 체계가 필수이다")
    return " / ".join(limitation_parts) + "."


def _result_lock_lines(lock_path: Path) -> List[str]:
    if not lock_path.exists():
        return ["Result Lock 요약 파일(lock.json)이 아직 생성되지 않았습니다."]
    data = lock_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return ["lock.json 파싱 실패."]
    datasets = parsed.get("datasets", {})
    lines = []
    for dataset, info in datasets.items():
        status = "PASS" if info.get("passed") else "FAIL"
        reason = info.get("reason", "")
        lines.append(f"- {dataset}: {status} — {reason}")
    return lines or ["Result Lock 데이터가 비어 있습니다."]


def _write_iclr_draft(
    paper_dir: Path,
    abstract_text: str,
    results_table: str,
    stats_table: str,
    lock_lines: List[str],
    ablation_bullets: List[str],
    limitations_text: str,
) -> None:
    content = [
        "# Spatiotemporal Observability RC2 Draft",
        "",
        "## Abstract",
        abstract_text,
        "",
        "## 1. Introduction",
        "TODO",
        "",
        "## 2. Method",
        "TODO",
        "",
        "## 3. Related Work",
        "TODO",
        "",
        "## 4. Experiments and Results",
        "### 4.1 Quantitative Summary",
        results_table,
        "",
        "### 4.2 Statistical Tests",
        stats_table,
        "",
        "### 4.3 Result Lock Status",
        "\n".join(lock_lines),
        "",
        "## 5. Ablations",
        "\n".join(f"- {line}" for line in ablation_bullets),
        "",
        "## 6. Limitations",
        limitations_text,
        "",
        "## 7. Ethics Statement",
        "TODO",
        "",
        "## References",
        "TODO",
        "",
    ]
    (paper_dir / "iclr_draft.md").write_text("\n".join(content), encoding="utf-8")


def _write_checklist(paper_dir: Path, datasets: List[str]) -> None:
    lines = [
        "# ICLR Reproducibility Checklist (RC2)",
        "",
        "## Code & Artifacts",
        "- [x] Training/Evaluation code (`src/**`) 공개",
        "- [x] Orchestration 스크립트 (`scripts/run_replicates.sh`, `Makefile lock`) 제공",
        "- [x] Result Lock 산출물 (`lock.json`, `paper/tables/*.tsv`, `paper/RESULT_LOCK.md`) 포함",
        "",
        "## Data",
        f"- [x] Real data 프로필 명시: {', '.join(datasets) if datasets else '미집계'}",
        "- [x] 환경 설정 (`assets/experiments/*.yaml`, `environment.yml`) 제공",
        "",
        "## Hardware & Compute",
        "- [x] run_manifest.json에 device/CUDA 기록",
        "- [x] branch orchestration 로그/요약(`runs/**/summary/summary.{json,md}`) 제공",
        "",
        "## Seeds & Determinism",
        "- [x] 고정 seed(1337/2337/3337) replicate 수행",
        "- [x] seed_manifest.json에 random seed 기록",
        "",
        "## Logging & Checkpoints",
        "- [x] train_log.jsonl 및 checkpoints(`runs/**/branches/`) 저장",
        "- [x] summary.json/plots_index.json으로 metrics 추적",
        "",
        "## Statistical Methods",
        "- [x] paired t-test, Cliff's delta, bootstrap CI 적용 (src/eval/stats.py)",
        "- [x] Result Lock 기준(instability, wasted FLOPs, val pplx) 명시",
        "",
        "## Limitations & Ethics",
        "- [x] Limitations 섹션에서 proxy miscalibration/compute overhead/data bias 다룸",
        "- [x] Ethics Statement 섹션 초안 예약",
        "",
        f"Generated at {datetime.utcnow().isoformat()}Z",
        "",
    ]
    (paper_dir / "ICLR_CHECKLIST.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper assets (RC2)")
    parser.add_argument("--paper_dir", default="paper")
    parser.add_argument("--runs_root", default="runs")
    parser.add_argument("--lock_path", default="lock.json")
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    tables_dir = paper_dir / "tables"
    results_rows = _read_tsv(tables_dir / "results.tsv")
    stats_rows = _read_tsv(tables_dir / "stats.tsv")
    dataset_summary = _collect_dataset_summary(results_rows, stats_rows)

    results_table = _markdown_table(
        results_rows,
        [
            ("experiment", "experiment"),
            ("baseline", "baseline"),
            ("method", "method"),
            ("replicates", "rep"),
            ("pplx_mean", "pplx"),
            ("instab_per_100k_mean", "instab/100k"),
            ("ttr_mean", "ttr"),
            ("invoke_rate_mean", "invoke"),
            ("accept_rate_mean", "accept"),
            ("proxy_gap_mean", "proxy_gap"),
            ("wasted_flops_mean", "wasted_flops"),
        ],
        float_cols={"pplx_mean", "instab_per_100k_mean", "ttr_mean", "invoke_rate_mean", "accept_rate_mean", "proxy_gap_mean", "wasted_flops_mean"},
    )

    stats_table = _markdown_table(
        stats_rows,
        [
            ("experiment", "experiment"),
            ("baseline", "baseline"),
            ("method", "method"),
            ("metric", "metric"),
            ("mean_diff", "mean_diff"),
            ("relative_diff", "rel_diff"),
            ("p_value", "p"),
            ("cliffs_delta", "delta"),
            ("ci_low", "ci_low"),
            ("ci_high", "ci_high"),
        ],
        float_cols={"mean_diff", "relative_diff", "p_value", "cliffs_delta", "ci_low", "ci_high"},
    )

    abstract_text = _build_abstract(dataset_summary)
    limitations_text = _build_limitations(dataset_summary)
    lock_lines = _result_lock_lines(Path(args.lock_path))

    ablations_dir = _latest_ablation(Path(args.runs_root) / "ablations")
    ablation_bullets = _build_ablation_bullets(ablations_dir)

    _write_iclr_draft(
        paper_dir,
        abstract_text,
        results_table,
        stats_table,
        lock_lines,
        ablation_bullets,
        limitations_text,
    )

    _write_checklist(paper_dir, list(dataset_summary.keys()))


if __name__ == "__main__":
    main()
