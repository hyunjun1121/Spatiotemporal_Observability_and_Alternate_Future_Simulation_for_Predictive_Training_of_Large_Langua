from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner."""

    parser = argparse.ArgumentParser(description="Spatiotemporal Observability experiment runner")
    parser.add_argument("--mode", default="baseline", choices=["baseline", "branch-test"], help="Run mode")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "hf"], help="Dataset mode")
    parser.add_argument("--real_data", default="off", choices=["wikitext", "c4", "off"], help="Select real dataset (off=synthetic)")
    parser.add_argument("--hf_name", default="wikitext-103", help="HF datasets identifier")
    parser.add_argument("--offline_data_dir", default=None, help="Path to offline shards (streaming)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--steps", type=int, default=400, help="Training steps")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--replicates", type=int, default=1, help="Number of replicate runs")
    parser.add_argument("--world_size", type=int, default=1, help="DDP world size metadata")
    parser.add_argument("--devices", default=None, help="CUDA device mask metadata")
    parser.add_argument("--max_concurrent", type=int, default=1, help="Scheduler max concurrency metadata")
    parser.add_argument("--resume_flag", action="store_true", help="Indicates the run was resumed by scheduler")
    parser.add_argument("--config", default=None, help="Path to experiment config YAML")
    parser.add_argument(
        "--baseline",
        default="none",
        choices=["none", "fixedlr", "hypergrad", "pbtlite", "zclip", "spamlite"],
        help="Baseline configuration",
    )
    parser.add_argument(
        "--method",
        default="C",
        choices=["A", "B", "C", "D"],
        help="Method variant (A:no-proxy, B:proxy-only, C:full, D:single-signal)",
    )
    parser.add_argument("--experiment_name", default="baseline", help="Experiment name prefix")
    parser.add_argument("--run_dir", default=None, help="Output root directory")
    parser.add_argument("--cooldown", type=int, default=300, help="Trigger cooldown steps")
    parser.add_argument("--epsilon_ucb", type=float, default=0.1, help="UCB1 epsilon exploration ratio")
    parser.add_argument("--ablation_mode", default="C", choices=["A", "B", "C", "D"], help="Legacy ablation flag")
    parser.add_argument("--ddp", type=int, default=1, help="torchrun world size")
    return parser.parse_args()
