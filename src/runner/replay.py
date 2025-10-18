from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.branch.actions import apply_action
from src.cp.checkpoint import load_state
from src.cp.state import restore_state
from src.data.datasets import build_dataloader
from src.train.build import build_model, build_optimizer, build_scheduler
from src.utils.config import load_yaml
from src.utils.seed import set_all_seeds


def _load_decision(decision_json: str) -> Dict[str, Any]:
    with open(decision_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _set_rng(state: Dict[str, Any]) -> None:
    import random

    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def _build_training_components(total_steps: int, precision: str) -> Any:
    state_schema = load_yaml("assets/state_schema_v1.yaml")
    model = build_model(state_schema.get("model", {}).get("arch", "gpt2-medium"), precision=precision)
    optimizer = build_optimizer(model, weight_decay=0.01, lr=3e-4)
    scheduler = build_scheduler(optimizer, state_schema.get("lr_scheduler", {}), total_steps=total_steps)
    return model, optimizer, scheduler


def _forward_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    batch,
    device: torch.device,
) -> Dict[str, float]:
    loss_fn = nn.CrossEntropyLoss()
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(x, return_attentions=False)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = math.sqrt(grad_norm)
    optimizer.step()
    scheduler.step()
    return {"loss": float(loss.item()), "grad_norm": float(grad_norm)}


def deterministic_replay(
    cp_path: str,
    decision_json: str,
    steps: int = 50,
    dataset: str = "synthetic",
    batch_size: int = 8,
    seq_len: int = 128,
) -> Dict[str, Any]:
    """체크포인트 복원 후 동일 액션을 재실행해 차이를 측정한다."""

    state = load_state(cp_path)
    decision = _load_decision(decision_json)

    total_steps = state.get("global_step", 0) + steps + 10
    precision = load_yaml("assets/state_schema_v1.yaml").get("model", {}).get("dtype", "bf16")
    model, optimizer, scheduler = _build_training_components(total_steps, precision)
    restore_state(model, optimizer, scheduler, state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _set_rng(state)
    noise_cfg = load_yaml("assets/noise_config.yaml")
    loader = build_dataloader(dataset, noise_cfg, batch_size=batch_size, seq_len=seq_len)
    iterator = iter(loader)

    accepted_action = decision.get("accepted_action") or {}
    apply_action({"action": accepted_action, "optimizer": optimizer})

    history: List[Dict[str, float]] = []
    for _ in range(steps):
        batch = next(iterator)
        metrics = _forward_step(model, optimizer, scheduler, batch, device)
        history.append(metrics)

    reference_runs = decision.get("verification_metrics", []) or decision.get("verify", [])
    reference_history = None
    for item in reference_runs:
        if item.get("accepted"):
            reference_history = item.get("history")
            break
    if reference_history is None and reference_runs:
        reference_history = reference_runs[0].get("history")
    if reference_history is None:
        raise ValueError("verification history not found in decision JSON")

    diffs = {"loss": 0.0, "grad_norm": 0.0}
    for new, old in zip(history, reference_history):
        diffs["loss"] = max(diffs["loss"], abs(new.get("loss", 0.0) - float(old.get("loss", 0.0))))
        diffs["grad_norm"] = max(diffs["grad_norm"], abs(new.get("grad_norm", 0.0) - float(old.get("grad_norm", 0.0))))

    tolerance = diffs["loss"] < 1e-5 and diffs["grad_norm"] < 1e-5
    return {
        "max_abs_diff_loss": diffs["loss"],
        "max_abs_diff_grad_norm": diffs["grad_norm"],
        "steps": steps,
        "tolerance_met": tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp_path", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    set_all_seeds(42)
    report = deterministic_replay(
        cp_path=args.cp_path,
        decision_json=args.decision,
        steps=args.steps,
        dataset=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    print(json.dumps(report, indent=2))
    print(
        "max|Δloss|={:.2e}, max|Δgrad_norm|={:.2e}, tolerance_met={}".format(
            report["max_abs_diff_loss"], report["max_abs_diff_grad_norm"], report["tolerance_met"]
        )
    )


if __name__ == "__main__":
    main()

