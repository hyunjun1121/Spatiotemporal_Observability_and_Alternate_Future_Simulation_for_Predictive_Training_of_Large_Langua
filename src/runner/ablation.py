from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def _collect_runs(root: Path) -> List[Path]:
    runs: List[Path] = []
    if not root.exists():
        return runs
    for child in root.iterdir():
        if child.is_dir():
            runs.append(child)
    return sorted(runs)


def _summarize_run(run_dir: Path) -> Dict[str, float]:
    log_file_jsonl = run_dir / "train_log.jsonl"
    metrics = {
        "loss_last": None,
        "instability_events": 0,
        "branch_invoke": 0,
        "accept_rate": 0.0,
    }
    accepted = 0
    triggered = 0
    steps = 0

    if log_file_jsonl.exists():
        with log_file_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                steps += 1
                metrics["loss_last"] = rec.get("loss")
                if rec.get("event_trigger_flag"):
                    metrics["instability_events"] += 1
                    triggered += 1
                branch = rec.get("branch", {}) or {}
                if branch.get("verified", 0) > 0:
                    metrics["branch_invoke"] += 1
                    if branch.get("accepted_action"):
                        accepted += 1
    metrics["accept_rate"] = (accepted / triggered) if triggered else 0.0
    metrics["steps"] = steps
    return metrics


def run(root: str, out: str) -> None:
    root_path = Path(root)
    runs = _collect_runs(root_path)
    rows = []
    for run_dir in runs:
        summary = _summarize_run(run_dir)
        rows.append({"run": run_dir.name, **summary})

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Markdown table
    headers = ["run", "loss_last", "instability_events", "branch_invoke", "accept_rate", "steps"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.root, args.out)


if __name__ == "__main__":
    main()

