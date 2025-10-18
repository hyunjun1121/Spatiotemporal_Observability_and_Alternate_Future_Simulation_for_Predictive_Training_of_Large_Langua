from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _extract_series(records: List[Dict[str, float]], key: str) -> List[float]:
    return [float(rec.get(key, 0.0)) for rec in records]


def generate_plots(
    run_dir: str,
    records: List[Dict[str, float]],
    top_actions: Sequence[Tuple[str, int]] | None = None,
    action_success: Dict[str, int] | None = None,
) -> List[str]:
    """Save key plots (loss, uncertainty, proxy gap, action bar chart) for a run."""

    if not records:
        return []

    steps = [rec.get("step", idx) for idx, rec in enumerate(records)]
    plot_dir = os.path.join(run_dir, "summary", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    paths: List[str] = []

    # Loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(steps, _extract_series(records, "loss"), label="loss")
    plt.plot(steps, _extract_series(records, "val_loss"), label="val_loss", alpha=0.7)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Loss vs Step")
    plt.legend()
    loss_path = os.path.join(plot_dir, "loss.png")
    plt.savefig(loss_path, dpi=120, bbox_inches="tight")
    plt.close()
    paths.append(os.path.relpath(loss_path, run_dir))

    # Uncertainty plot
    plt.figure(figsize=(8, 4))
    plt.plot(steps, _extract_series(records, "uncertainty_index"), label="uncertainty_index", color="tab:orange")
    trigger_steps = [rec.get("step", 0) for rec in records if rec.get("triggered")]
    trigger_values = [rec.get("uncertainty_index", 0.0) for rec in records if rec.get("triggered")]
    if trigger_steps:
        plt.scatter(trigger_steps, trigger_values, color="red", marker="x", label="trigger")
    plt.xlabel("step")
    plt.ylabel("uncertainty")
    plt.title("Uncertainty Index")
    plt.legend()
    unc_path = os.path.join(plot_dir, "uncertainty.png")
    plt.savefig(unc_path, dpi=120, bbox_inches="tight")
    plt.close()
    paths.append(os.path.relpath(unc_path, run_dir))

    # Proxy-real gap plot
    plt.figure(figsize=(8, 4))
    plt.plot(steps, _extract_series(records, "proxy_real_gap"), label="proxy_real_gap", color="tab:green")
    plt.xlabel("step")
    plt.ylabel("gap")
    plt.title("Proxy Real Gap")
    plt.legend()
    gap_path = os.path.join(plot_dir, "proxy_gap.png")
    plt.savefig(gap_path, dpi=120, bbox_inches="tight")
    plt.close()
    paths.append(os.path.relpath(gap_path, run_dir))

    if top_actions:
        labels = [json.loads(action) if action.startswith("{") else action for action, _ in top_actions]
        labels = [str(label) for label in labels]
        counts = [count for _, count in top_actions]
        success_rates = []
        for action, count in top_actions:
            success = 0
            if action_success and count:
                success = action_success.get(action, 0) / count
            success_rates.append(success)

        plt.figure(figsize=(8, 4))
        plt.bar(range(len(labels)), counts, color="tab:blue", alpha=0.7, label="count")
        plt.ylabel("count")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.title("Top Actions (count & success rate)")
        plt.twinx()
        plt.plot(range(len(labels)), success_rates, color="tab:red", marker="o", label="success rate")
        plt.ylabel("success rate")
        plt.ylim(0, 1)
        plt.tight_layout()
        bar_path = os.path.join(plot_dir, "actions.png")
        plt.savefig(bar_path, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(os.path.relpath(bar_path, run_dir))

    return paths


def plot_metric_ci_bars(records: List[Dict[str, Any]], out_path: Path, title: str, ylabel: str) -> None:
    """Generate grouped bar plot with confidence intervals for aggregated metrics."""

    if not records:
        return

    experiments = sorted({rec["experiment"] for rec in records})
    methods = sorted({rec["method"] for rec in records})
    if not experiments or not methods:
        return

    x = np.arange(len(experiments))
    width = min(0.8 / max(1, len(methods)), 0.35)

    plt.figure(figsize=(8, 4))
    for idx, method in enumerate(methods):
        means = []
        lower_err: List[float] = []
        upper_err: List[float] = []
        for exp in experiments:
            rec = next((item for item in records if item["experiment"] == exp and item["method"] == method), None)
            if rec is None or not np.isfinite(rec.get("mean", np.nan)):
                means.append(np.nan)
                lower_err.append(0.0)
                upper_err.append(0.0)
                continue
            mean = float(rec["mean"])
            ci_low = rec.get("ci_low", mean)
            ci_high = rec.get("ci_high", mean)
            lower = max(mean - ci_low, 0.0) if np.isfinite(ci_low) else 0.0
            upper = max(ci_high - mean, 0.0) if np.isfinite(ci_high) else 0.0
            means.append(mean)
            lower_err.append(lower)
            upper_err.append(upper)
        offsets = x + (idx - (len(methods) - 1) / 2) * width
        plt.bar(offsets, means, width=width, label=f"method {method}")
        plt.errorbar(offsets, means, yerr=[lower_err, upper_err], fmt="none", ecolor="black", capsize=4)

    plt.xticks(x, experiments, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
