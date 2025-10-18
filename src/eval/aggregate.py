from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from datetime import datetime
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from src.eval.cost import estimate_co2e, estimate_energy, estimate_flops
from src.eval.plots import generate_plots, plot_metric_ci_bars
from src.eval.stats import bootstrap_ci, lock_judgement


SUMMARY_FIELDS = [
    "val_pplx",
    "instability_events_per_100k",
    "time_to_recover_mean",
    "branch_invoke_rate",
    "accept_rate",
    "proxy_real_gap_mean",
    "wasted_flops_est_total",
]

ALIAS_MAP = {
    "val_pplx": ("pplx", "Validation Perplexity"),
    "instability_events_per_100k": ("instab_per_100k", "Instability Events per 100k"),
    "time_to_recover_mean": ("ttr", "Time To Recover"),
    "branch_invoke_rate": ("invoke_rate", "Branch Invoke Rate"),
    "accept_rate": ("accept_rate", "Accept Rate"),
    "proxy_real_gap_mean": ("proxy_gap", "Proxy Gap"),
    "wasted_flops_est_total": ("wasted_flops", "Wasted FLOPs Estimate"),
}

PLOT_TARGETS = [
    ("val_pplx", "pplx", "Validation Perplexity"),
    ("instability_events_per_100k", "instab_per_100k", "Instability Events per 100k"),
    ("wasted_flops_est_total", "wasted_flops", "Wasted FLOPs Estimate"),
]


def _load_config_snapshot(run_dir: Path) -> Dict[str, Any]:
    snapshot_yaml = run_dir / "config_snapshot.yaml"
    snapshot_json = run_dir / "config_snapshot.json"
    if snapshot_yaml.exists():
        try:
            import yaml  # type: ignore

            return yaml.safe_load(snapshot_yaml.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    if snapshot_json.exists():
        return json.loads(snapshot_json.read_text(encoding="utf-8"))
    return {}


def _load_jsonl(path: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _time_to_recover(records: List[Dict[str, float]]) -> List[int]:
    recoveries: List[int] = []
    pending = None
    for rec in records:
        if rec.get("triggered"):
            pending = rec.get("step")
        elif pending is not None and not rec.get("triggered"):
            recoveries.append(int(rec.get("step", 0)) - int(pending))
            pending = None
    return recoveries


def aggregate_run(run_dir: str) -> Dict[str, float]:
    run_path = Path(run_dir)
    log_path = run_path / "train_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    records = _load_jsonl(log_path)
    manifest = _load_manifest(run_path)
    config_snapshot = _load_config_snapshot(run_path)
    steps = len(records)

    instability_events = sum(1 for rec in records if rec.get("event_trigger_flag"))
    instability_per_100k = 1e5 * instability_events / max(1, steps)

    branch_invocations = sum(1 for rec in records if (rec.get("branch") or {}).get("candidate_count", 0) > 0)
    branch_invoke_rate = branch_invocations / max(1, steps)

    accepted = sum(1 for rec in records if (rec.get("branch") or {}).get("accepted_action"))
    accept_rate = accepted / max(1, instability_events)

    proxy_gaps = [rec.get("proxy_real_gap", 0.0) for rec in records]
    proxy_gap_mean = float(statistics.fmean(proxy_gaps)) if proxy_gaps else 0.0
    proxy_gap_median = float(statistics.median(proxy_gaps)) if proxy_gaps else 0.0

    recoveries = _time_to_recover(records)
    recovery_mean = float(statistics.fmean(recoveries)) if recoveries else 0.0
    recovery_median = float(statistics.median(recoveries)) if recoveries else 0.0

    wasted_total = float(records[-1].get("wasted_flops_est", 0.0)) if records else 0.0

    val_losses = [float(rec.get("val_loss", rec.get("loss", 0.0))) for rec in records]
    if val_losses:
        val_loss_mean = float(statistics.fmean(val_losses))
        val_pplx = float(math.exp(val_loss_mean))
    else:
        val_loss_mean = float("nan")
        val_pplx = float("nan")

    timeline = [
        {
            "step": rec.get("step", 0),
            "triggered": bool(rec.get("triggered")),
            "accepted": bool((rec.get("branch") or {}).get("accepted_action")),
        }
        for rec in records
        if rec.get("triggered") or (rec.get("branch") or {}).get("accepted_action")
    ]

    action_counts: Counter[str] = Counter()
    action_success: Counter[str] = Counter()
    for rec in records:
        action = rec.get("action_applied")
        if action:
            key = json.dumps(action, sort_keys=True)
            action_counts[key] += 1
            if (rec.get("branch") or {}).get("accepted_action"):
                action_success[key] += 1

    top_actions = action_counts.most_common(3)

    run_cfg = config_snapshot.get("run", {})
    model_cfg = config_snapshot.get("model", {"hidden_size": 1024, "num_layers": 24})
    seq_est = int(run_cfg.get("seq_len", 512))
    batch_est = int(run_cfg.get("batch_size", manifest.get("batch_size", 8)))
    steps_est = int(run_cfg.get("steps", steps))
    flops_est = estimate_flops(model_cfg, steps_est, seq_est, batch_est)
    energy_kj = estimate_energy(flops_est)
    co2e = estimate_co2e(energy_kj)

    summary = {
        "steps": steps,
        "instability_events": instability_events,
        "instability_events_per_100k": instability_per_100k,
        "time_to_recover_mean": recovery_mean,
        "time_to_recover_median": recovery_median,
        "branch_invoke_rate": branch_invoke_rate,
        "accept_rate": accept_rate,
        "proxy_real_gap_mean": proxy_gap_mean,
        "proxy_real_gap_median": proxy_gap_median,
        "wasted_flops_est_total": wasted_total,
        "val_loss_mean": val_loss_mean,
        "val_pplx": val_pplx,
        "flops_est": flops_est,
        "energy_kj": energy_kj,
        "co2e_tons": co2e,
    }

    summary_dir = run_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with (summary_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({**summary, "timeline": timeline, "top_actions": top_actions}, f, ensure_ascii=False, indent=2)

    lines = [f"# Summary for {manifest.get('experiment', 'unknown')} / {manifest.get('run_id', 'unknown')}", ""]

    meta_rows = {
        "created_utc": manifest.get("created_utc"),
        "device": manifest.get("device"),
        "cuda_available": manifest.get("cuda_available"),
        "python_version": manifest.get("python_version"),
        "torch_version": manifest.get("torch_version"),
        "steps": steps,
        "baseline": manifest.get("baseline") or run_cfg.get("baseline"),
        "method": manifest.get("method") or run_cfg.get("method"),
        "real_data": manifest.get("real_data") or run_cfg.get("real_data"),
        "config_path": manifest.get("config_path"),
        "batch_size": batch_est,
    }

    lines.append("| Metadata | Value |")
    lines.append("|---|---|")
    for key, value in meta_rows.items():
        lines.append(f"| {key} | {value} |")

    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")

    if timeline:
        lines.append("")
        lines.append("## Trigger / Accept Timeline")
        lines.append("| step | triggered | accepted |")
        lines.append("|---|---|---|")
        for item in timeline:
            lines.append(f"| {item['step']} | {item['triggered']} | {item['accepted']} |")

    if top_actions:
        lines.append("")
        lines.append("## Top Actions")
        lines.append("| action | count | success_rate |")
        lines.append("|---|---|---|")
        for action, count in top_actions:
            success = action_success[action] / max(1, count)
            lines.append(f"| `{action}` | {count} | {success:.2f} |")

    with (summary_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    plot_paths = generate_plots(run_dir, records, top_actions, action_success)
    if plot_paths:
        with (summary_dir / "plots_index.json").open("w", encoding="utf-8") as f:
            json.dump(plot_paths, f, ensure_ascii=False, indent=2)

    return summary


def _collect_run_records(runs_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for manifest_path in runs_root.rglob("run_manifest.json"):
        run_dir = manifest_path.parent
        try:
            summary = aggregate_run(str(run_dir))
        except FileNotFoundError:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed = int(manifest.get("seed", 0))
        records.append({"run_dir": run_dir, "manifest": manifest, "summary": summary, "seed": seed})
    return records


def _copy_run_figures(records: List[Dict[str, Any]], figs_dir: Path) -> List[str]:
    outputs: List[str] = []
    for rec in records:
        run_dir: Path = rec["run_dir"]
        plots_index = run_dir / "summary" / "plots_index.json"
        if not plots_index.exists():
            continue
        try:
            files = json.loads(plots_index.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for rel in files:
            src = run_dir / rel
            if not src.exists():
                continue
            dst = figs_dir / f"{run_dir.name}_{Path(rel).name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            outputs.append(str(dst))
    return outputs


def _group_runs(records: List[Dict[str, Any]], group_cols: List[str]) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for rec in records:
        manifest = rec["manifest"]
        key = tuple(str(manifest.get(col, "")) for col in group_cols)
        group = groups.setdefault(key, {"meta": {}, "runs": []})
        group["runs"].append(rec)
        for col in group_cols:
            if col not in group["meta"]:
                group["meta"][col] = manifest.get(col, "")
    return groups


def _prepare_group_data(groups: Dict[Tuple[str, ...], Dict[str, Any]]) -> None:
    for group in groups.values():
        seed_map: Dict[int, Dict[str, Any]] = {}
        for rec in group["runs"]:
            seed_map[int(rec["seed"])] = rec
        seeds = sorted(seed_map.keys())
        metrics: Dict[str, np.ndarray] = {}
        aggregates: Dict[str, Dict[str, float]] = {}
        for field in SUMMARY_FIELDS:
            values = []
            for seed in seeds:
                summary = seed_map[seed]["summary"]
                values.append(float(summary.get(field, float("nan"))))
            arr = np.asarray(values, dtype=np.float64)
            metrics[field] = arr
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                mean = float("nan")
                std = float("nan")
                ci_low = float("nan")
                ci_high = float("nan")
            else:
                mean = float(valid.mean())
                std = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
                ci_low, ci_high = bootstrap_ci(valid)
            aggregates[field] = {
                "mean": mean,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "valid": int(valid.size),
            }
        group["seeds"] = seeds
        group["seed_index"] = {seed: idx for idx, seed in enumerate(seeds)}
        group["metrics"] = metrics
        group["aggregates"] = aggregates
        group["replicates"] = len(seeds)


def _format_float(value: float, precision: int = 4) -> str:
    if value != value or value is None:  # NaN check
        return "nan"
    fmt = f"{{0:.{precision}f}}"
    return fmt.format(value)


def _write_results_tsv(rows: List[Dict[str, Any]], out_path: Path, group_cols: List[str]) -> None:
    headers: List[str] = list(group_cols)
    headers.append("replicates")
    for field, (alias, _) in ALIAS_MAP.items():
        headers.extend(
            [
                f"{alias}_mean",
                f"{alias}_std",
                f"{alias}_ci_low",
                f"{alias}_ci_high",
            ]
        )
    lines = ["\t".join(headers)]
    for row in rows:
        parts: List[str] = []
        for col in group_cols:
            parts.append(str(row.get(col, "")))
        parts.append(str(row.get("replicates", 0)))
        for field, (alias, _) in ALIAS_MAP.items():
            agg = row.get("aggregates", {}).get(field, {})
            parts.append(_format_float(agg.get("mean", float("nan")), 4))
            parts.append(_format_float(agg.get("std", float("nan")), 4))
            parts.append(_format_float(agg.get("ci_low", float("nan")), 4))
            parts.append(_format_float(agg.get("ci_high", float("nan")), 4))
        lines.append("\t".join(parts))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stats_tsv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    headers = [
        "experiment",
        "baseline",
        "method",
        "metric",
        "baseline_mean",
        "method_mean",
        "mean_diff",
        "relative_diff",
        "p_value",
        "cliffs_delta",
        "ci_low",
        "ci_high",
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        parts = [
            str(row.get("experiment", "")),
            str(row.get("baseline", "")),
            str(row.get("method", "")),
            str(row.get("metric", "")),
            _format_float(row.get("baseline_mean", float("nan")), 4),
            _format_float(row.get("method_mean", float("nan")), 4),
            _format_float(row.get("mean_diff", float("nan")), 4),
            _format_float(row.get("relative_diff", float("nan")), 4),
            _format_float(row.get("p_value", float("nan")), 6),
            _format_float(row.get("cliffs_delta", float("nan")), 4),
            _format_float(row.get("ci_low", float("nan")), 4),
            _format_float(row.get("ci_high", float("nan")), 4),
        ]
        lines.append("\t".join(parts))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_lock_artifacts(lock_summary: Dict[str, Any], lock_json: Path, lock_md: Path) -> None:
    payload = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "criteria": {
            "instability_events_per_100k": "-30% 이상 감소",
            "wasted_flops_est_total": "-20% 이상 감소",
            "val_pplx": "±0.5% 이내 변동",
            "significance": {
                "paired_t_test": "p < 0.05",
                "cliffs_delta": "|δ| ≥ 0.33",
                "bootstrap_ci": "95% CI",
            },
        },
        "datasets": lock_summary,
    }
    lock_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    lines = ["# Result Lock Summary", ""]
    for dataset, info in lock_summary.items():
        status = "PASS" if info.get("passed") else "FAIL"
        reason = info.get("reason", "")
        lines.append(f"- **{dataset}** - {status}: {reason}")
        metrics = info.get("metrics", {})
        for metric, stats in metrics.items():
            base_mean = stats.get("baseline_mean")
            method_mean = stats.get("method_mean")
            rel = stats.get("relative_diff")
            delta = stats.get("cliffs_delta")
            p_val = stats.get("p_value")
            ci_low = stats.get("ci_low")
            ci_high = stats.get("ci_high")
            improvement = stats.get("improvement")
            detail_parts = [
                f"baseline={base_mean:.4f}" if base_mean == base_mean else "baseline=nan",
                f"method={method_mean:.4f}" if method_mean == method_mean else "method=nan",
            ]
            if improvement is not None and improvement == improvement:
                detail_parts.append(f"Δ={improvement:+.3%}")
            elif rel is not None and rel == rel:
                detail_parts.append(f"Δ={rel:+.3%}")
            if p_val is not None and p_val == p_val:
                detail_parts.append(f"p={p_val:.3g}")
            if delta is not None and delta == delta:
                detail_parts.append(f"δ={delta:.3f}")
            if ci_low is not None and ci_high is not None and ci_low == ci_low and ci_high == ci_high:
                detail_parts.append(f"CI=[{ci_low:.4f},{ci_high:.4f}]")
            lines.append(f"  - {metric}: " + ", ".join(detail_parts))
        lines.append("")
    lock_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _get_group(
    groups: Dict[Tuple[str, ...], Dict[str, Any]],
    group_cols: List[str],
    query: Dict[str, str],
) -> Dict[str, Any] | None:
    key = []
    for col in group_cols:
        key.append(str(query.get(col, "")))
    key_tuple = tuple(key)
    return groups.get(key_tuple)


def _generate_ci_plots(
    groups: Dict[Tuple[str, ...], Dict[str, Any]],
    group_cols: List[str],
    figs_dir: Path,
) -> List[str]:
    outputs: List[str] = []
    if not {"experiment", "baseline", "method"}.issubset(group_cols):
        return outputs
    col_index = {col: idx for idx, col in enumerate(group_cols)}
    experiments = sorted({group["meta"].get("experiment", "") for group in groups.values()})

    for summary_key, alias, title in PLOT_TARGETS:
        records: List[Dict[str, Any]] = []
        for exp in experiments:
            for method in ["A", "C"]:
                query = {
                    "experiment": exp,
                    "baseline": "fixedlr",
                    "method": method,
                }
                group = _get_group(groups, group_cols, query)
                if not group:
                    continue
                agg = group["aggregates"].get(summary_key, {})
                records.append(
                    {
                        "experiment": exp,
                        "method": method,
                        "mean": agg.get("mean", float("nan")),
                        "ci_low": agg.get("ci_low", float("nan")),
                        "ci_high": agg.get("ci_high", float("nan")),
                    }
                )
        if not records:
            continue
        out_path = figs_dir / f"bar_{alias}.png"
        plot_metric_ci_bars(records, out_path, title=title, ylabel=title)
        outputs.append(str(out_path))
    return outputs


def _build_results_rows(groups: Dict[Tuple[str, ...], Dict[str, Any]], group_cols: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, group in groups.items():
        row: Dict[str, Any] = {"aggregates": group["aggregates"], "replicates": group.get("replicates", 0)}
        for idx, col in enumerate(group_cols):
            row[col] = key[idx]
        rows.append(row)
    return rows


def _build_stats_and_lock(
    groups: Dict[Tuple[str, ...], Dict[str, Any]],
    group_cols: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stats_rows: List[Dict[str, Any]] = []
    lock_summary: Dict[str, Any] = {}
    if not {"experiment", "baseline", "method"}.issubset(group_cols):
        return stats_rows, lock_summary

    experiments = sorted({group["meta"].get("experiment", "") for group in groups.values()})
    baselines = sorted({group["meta"].get("baseline", "") for group in groups.values()})

    for exp in experiments:
        control = _get_group(groups, group_cols, {"experiment": exp, "baseline": "fixedlr", "method": "A"})
        treatment = _get_group(groups, group_cols, {"experiment": exp, "baseline": "fixedlr", "method": "C"})
        if control and treatment:
            passed, reason, details = lock_judgement(control, treatment)
            lock_summary[exp] = {
                "passed": passed,
                "reason": reason,
                "metrics": details,
                "replicates": min(control.get("replicates", 0), treatment.get("replicates", 0)),
            }
            for metric, info in details.items():
                stats_rows.append(
                    {
                        "experiment": exp,
                        "baseline": "fixedlr",
                        "method": "C",
                        "metric": metric,
                        "baseline_mean": info.get("baseline_mean", float("nan")),
                        "method_mean": info.get("method_mean", float("nan")),
                        "mean_diff": info.get("mean_diff", float("nan")),
                        "relative_diff": info.get("relative_diff", info.get("improvement", float("nan"))),
                        "p_value": info.get("p_value", float("nan")),
                        "cliffs_delta": info.get("cliffs_delta", float("nan")),
                        "ci_low": info.get("ci_low", float("nan")),
                        "ci_high": info.get("ci_high", float("nan")),
                    }
                )
        else:
            lock_summary[exp] = {
                "passed": False,
                "reason": "baseline 또는 method C 데이터가 부족합니다.",
                "metrics": {},
                "replicates": 0,
            }

        for baseline in baselines:
            if baseline == "fixedlr":
                continue
            control_group = _get_group(groups, group_cols, {"experiment": exp, "baseline": baseline, "method": "A"})
            method_c_group = _get_group(groups, group_cols, {"experiment": exp, "baseline": baseline, "method": "C"})
            if not control_group or not method_c_group:
                continue
            _, _, details = lock_judgement(control_group, method_c_group)
            for metric, info in details.items():
                stats_rows.append(
                    {
                        "experiment": exp,
                        "baseline": baseline,
                        "method": "C",
                        "metric": metric,
                        "baseline_mean": info.get("baseline_mean", float("nan")),
                        "method_mean": info.get("method_mean", float("nan")),
                        "mean_diff": info.get("mean_diff", float("nan")),
                        "relative_diff": info.get("relative_diff", info.get("improvement", float("nan"))),
                        "p_value": info.get("p_value", float("nan")),
                        "cliffs_delta": info.get("cliffs_delta", float("nan")),
                        "ci_low": info.get("ci_low", float("nan")),
                        "ci_high": info.get("ci_high", float("nan")),
                    }
                )

    return stats_rows, lock_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", help="Aggregate a single run directory")
    parser.add_argument("--runs_root", help="Root directory containing runs for aggregation")
    parser.add_argument("--paper_dir", default="paper", help="Directory to write paper assets")
    parser.add_argument("--groupby", default="experiment,baseline,method", help="Comma-separated manifest keys for grouping")
    parser.add_argument("--lock_json", default="lock.json", help="Path to write lock summary JSON")
    args = parser.parse_args()

    if args.run_dir:
        aggregate_run(args.run_dir)
        return

    if not args.runs_root:
        parser.error("--runs_root is required when --run_dir is not provided")

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")

    paper_dir = Path(args.paper_dir)
    tables_dir = paper_dir / "tables"
    figs_dir = paper_dir / "figs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    group_cols = [col.strip() for col in args.groupby.split(",") if col.strip()]
    records = _collect_run_records(runs_root)
    if not records:
        print("No runs discovered under the specified root; skipping aggregation.")
        return

    groups = _group_runs(records, group_cols)
    _prepare_group_data(groups)

    results_rows = _build_results_rows(groups, group_cols)
    _write_results_tsv(results_rows, tables_dir / "results.tsv", group_cols)

    stats_rows, lock_summary = _build_stats_and_lock(groups, group_cols)
    _write_stats_tsv(stats_rows, tables_dir / "stats.tsv")
    _write_lock_artifacts(lock_summary, Path(args.lock_json), paper_dir / "RESULT_LOCK.md")

    copied_plots = _copy_run_figures(records, figs_dir)
    plot_paths = _generate_ci_plots(groups, group_cols, figs_dir)
    for dataset, info in lock_summary.items():
        status = "PASS" if info.get("passed") else "FAIL"
        print(f"[Result Lock] {dataset}: {status} - {info.get('reason', '')}")
    if copied_plots:
        print(f"Copied {len(copied_plots)} run plots into {figs_dir}")
    if plot_paths:
        print(f"Generated CI bar plots: {', '.join(plot_paths)}")


if __name__ == "__main__":
    main()
