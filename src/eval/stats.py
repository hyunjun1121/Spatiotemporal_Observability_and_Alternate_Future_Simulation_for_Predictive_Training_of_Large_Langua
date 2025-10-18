from __future__ import annotations

from math import sqrt
from typing import Dict, Iterable, List, Tuple

import numpy as np


def paired_ttest(a: Iterable[float], b: Iterable[float]) -> Tuple[float, float]:
    arr_a = np.asarray(list(a), dtype=np.float64)
    arr_b = np.asarray(list(b), dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError("paired_ttest requires equal length arrays")
    diff = arr_a - arr_b
    n = diff.size
    if n < 2:
        return float("nan"), float("nan")
    mean = diff.mean()
    std = diff.std(ddof=1)
    t_stat = mean / (std / sqrt(n)) if std > 0 else float("inf")
    # two-sided p-value via survival function approximation
    try:
        from scipy.stats import t as student_t  # type: ignore

        p_value = 2 * student_t.sf(abs(t_stat), df=n - 1)
    except Exception:
        # 정규 근사 fallback
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    return float(t_stat), float(p_value)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    arr_a = np.asarray(list(a), dtype=np.float64)
    arr_b = np.asarray(list(b), dtype=np.float64)
    greater = 0
    lesser = 0
    for x in arr_a:
        greater += np.sum(x > arr_b)
        lesser += np.sum(x < arr_b)
    n1 = arr_a.size
    n2 = arr_b.size
    if n1 == 0 or n2 == 0:
        return 0.0
    return (greater - lesser) / (n1 * n2)


def bootstrap_ci(values: Iterable[float], *, n: int = 10000, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        val = float(arr[0])
        return val, val
    rng = np.random.default_rng(42)
    samples = [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n)]
    lower = np.percentile(samples, 100 * (alpha / 2))
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def _aligned_metric(
    group: Dict[str, np.ndarray],
    metric: str,
    seeds: List[int],
) -> np.ndarray:
    metrics = group.get("metrics", {})
    seed_index = group.get("seed_index", {})
    if metric not in metrics:
        return np.asarray([], dtype=np.float64)
    values = metrics[metric]
    aligned = [values[seed_index[seed]] for seed in seeds if seed in seed_index]
    return np.asarray(aligned, dtype=np.float64)


def lock_judgement(
    baseline_group: Dict[str, np.ndarray],
    method_group: Dict[str, np.ndarray],
) -> Tuple[bool, str, Dict[str, Dict[str, float]]]:
    seeds_a = baseline_group.get("seeds", [])
    seeds_b = method_group.get("seeds", [])
    shared = sorted(set(seeds_a) & set(seeds_b))
    if len(shared) < 3:
        return False, "매칭된 seed가 3개 미만입니다.", {}

    metrics_of_interest = {
        "val_pplx": "val_pplx",
        "instability_events_per_100k": "instability_events_per_100k",
        "wasted_flops_est": "wasted_flops_est_total",
    }
    optional_metrics = {
        "time_to_recover": "time_to_recover_mean",
        "branch_invoke_rate": "branch_invoke_rate",
        "accept_rate": "accept_rate",
        "proxy_real_gap": "proxy_real_gap_mean",
    }

    details: Dict[str, Dict[str, float]] = {}
    messages: List[str] = []

    # Primary criteria metrics
    results: Dict[str, bool] = {}
    for metric, summary_key in metrics_of_interest.items():
        base_vals = _aligned_metric(baseline_group, summary_key, shared)
        method_vals = _aligned_metric(method_group, summary_key, shared)
        if base_vals.size != method_vals.size or base_vals.size == 0:
            return False, f"{metric} metric 데이터가 충분하지 않습니다.", {}

        base_mean = float(base_vals.mean())
        method_mean = float(method_vals.mean())
        diff = method_vals - base_vals
        t_stat, p_value = paired_ttest(method_vals, base_vals)
        delta = cliffs_delta(method_vals, base_vals)
        ci_low, ci_high = bootstrap_ci(diff)

        entry = {
            "baseline_mean": base_mean,
            "method_mean": method_mean,
            "mean_diff": method_mean - base_mean,
            "p_value": p_value,
            "cliffs_delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

        if base_mean != 0.0:
            entry["relative_diff"] = (method_mean - base_mean) / base_mean
        else:
            entry["relative_diff"] = float("nan")

        details[metric] = entry

        if metric == "val_pplx":
            rel = entry["relative_diff"]
            cond = abs(rel) <= 0.005 if np.isfinite(rel) else False
            messages.append(
                f"val_pplx Δ {rel:+.3%} (허용 ±0.5%) — {'충족' if cond else '미달'}"
            )
            results[metric] = cond
        elif metric == "instability_events_per_100k":
            improvement = (base_mean - method_mean) / base_mean if base_mean else float("-inf")
            cond = (
                improvement >= 0.30
                and p_value < 0.05
                and delta <= -0.33
                and ci_high <= 0.0
            )
            messages.append(
                f"instability_events_per_100k 개선 {improvement:+.3%} (요구 ≥30%), p={p_value:.3g}, δ={delta:.3f}, CI=[{ci_low:.4f},{ci_high:.4f}] — {'충족' if cond else '미달'}"
            )
            entry["improvement"] = improvement
            results[metric] = cond
        elif metric == "wasted_flops_est":
            improvement = (base_mean - method_mean) / base_mean if base_mean else float("-inf")
            cond = (
                improvement >= 0.20
                and p_value < 0.05
                and delta <= -0.33
                and ci_high <= 0.0
            )
            messages.append(
                f"wasted_flops_est 개선 {improvement:+.3%} (요구 ≥20%), p={p_value:.3g}, δ={delta:.3f}, CI=[{ci_low:.4f},{ci_high:.4f}] — {'충족' if cond else '미달'}"
            )
            entry["improvement"] = improvement
            results[metric] = cond

    # Optional metrics stored for reporting
    for metric, summary_key in optional_metrics.items():
        base_vals = _aligned_metric(baseline_group, summary_key, shared)
        method_vals = _aligned_metric(method_group, summary_key, shared)
        if base_vals.size == 0 or method_vals.size == 0:
            continue
        diff = method_vals - base_vals
        _, p_value = paired_ttest(method_vals, base_vals)
        delta = cliffs_delta(method_vals, base_vals)
        ci_low, ci_high = bootstrap_ci(diff)
        details[metric] = {
            "baseline_mean": float(base_vals.mean()),
            "method_mean": float(method_vals.mean()),
            "mean_diff": float(diff.mean()),
            "p_value": p_value,
            "cliffs_delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    passed = all(results.get(metric, False) for metric in metrics_of_interest.keys())
    reason = " / ".join(messages)
    return passed, reason, details
