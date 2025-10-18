from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.branch.orchestrator import sample_actions, score_candidates
from src.cli import parse_args
from src.train.loop import train_baseline
from src.utils.config import load_yaml
from src.utils.io import ensure_dir


def _apply_config_overrides(args: Any, config_path: str | None) -> Dict[str, Any]:
    """Load experiment config YAML and override argparse namespace."""

    if not config_path:
        return {}

    config_data = load_yaml(config_path)
    if not config_data:
        return {}

    if "dataset" in config_data:
        args.dataset = config_data["dataset"]
    if "real_data" in config_data:
        args.real_data = config_data["real_data"]
    if "hf_name" in config_data:
        args.hf_name = config_data["hf_name"]
    if "seq_len" in config_data:
        args.seq_len = int(config_data["seq_len"])
    if "batch_size" in config_data:
        args.batch_size = int(config_data["batch_size"])
    if "steps" in config_data:
        args.steps = int(config_data["steps"])
    if "replicates" in config_data:
        args.replicates = int(config_data["replicates"])
    if "cooldown" in config_data:
        args.cooldown = int(config_data["cooldown"])
    if "epsilon_ucb" in config_data:
        args.epsilon_ucb = float(config_data["epsilon_ucb"])
    if "world_size" in config_data:
        args.world_size = int(config_data["world_size"])
    if "devices" in config_data:
        args.devices = str(config_data["devices"])
    if "max_concurrent" in config_data:
        args.max_concurrent = int(config_data["max_concurrent"])
    if "resume_flag" in config_data:
        args.resume_flag = bool(config_data["resume_flag"])
    if "offline_data_dir" in config_data and config_data["offline_data_dir"]:
        args.offline_data_dir = str(config_data["offline_data_dir"])
    if "baseline" in config_data:
        args.baseline = config_data["baseline"]
    if "method" in config_data:
        args.method = config_data["method"]
    if "experiment_name" in config_data:
        args.experiment_name = str(config_data["experiment_name"])
    elif "experiment" in config_data:
        args.experiment_name = str(config_data["experiment"])
    elif config_path:
        args.experiment_name = Path(config_path).stem

    return config_data


def main() -> None:
    """CLI entrypoint for baseline training or branch-test dry-run."""

    args = parse_args()
    config_path = args.config
    experiment_config = _apply_config_overrides(args, config_path)

    if args.mode == "baseline":
        if args.replicates == 3:
            seeds = [1337, 2337, 3337]
        else:
            seeds = [args.seed + i for i in range(args.replicates)]
        for idx, run_seed in enumerate(seeds):
            train_baseline(
                dataset_mode=args.dataset,
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                seed=run_seed,
                precision="bf16",
                experiment_name=args.experiment_name,
                run_dir=args.run_dir,
                cooldown=args.cooldown,
                epsilon_ucb=args.epsilon_ucb,
                ablation_mode=args.method,
                real_data=args.real_data,
                hf_name=args.hf_name,
                offline_data_dir=args.offline_data_dir,
                baseline=args.baseline,
                method=args.method,
                replicate_id=idx,
                config_path=config_path,
                experiment_config=experiment_config,
                devices=args.devices,
                world_size=args.world_size,
                max_concurrent=args.max_concurrent,
                resume_flag=args.resume_flag,
            )
    else:
        space = load_yaml("assets/hparam_space.yaml")
        policy = load_yaml("assets/branch_policy.yaml")
        actions = sample_actions(space, int(space.get("sample_size", 128)))[0]
        dummy_preds = [{"loss_at_N": 1.0, "risk_quantile": 0.1, "recovery_time": 10.0} for _ in actions]
        scores = score_candidates(dummy_preds, policy)
        ensure_dir("branches/test")
        with open("branches/test/dry_run.json", "w", encoding="utf-8") as f:
            json.dump({"count": len(actions), "top_score": float(scores.max())}, f, ensure_ascii=False, indent=2)
        print("Branch-test dry-run complete.")


if __name__ == "__main__":
    main()
