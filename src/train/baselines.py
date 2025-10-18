from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class BaselineState:
    method: str
    best_loss: float = float("inf")
    hyper_beta: float = 0.01
    last_grad: float = 0.0


def apply_fixed_cosine(scheduler) -> None:
    if scheduler is not None and hasattr(scheduler, "step"):
        scheduler.step()


def apply_hypergradient(optimizer: torch.optim.Optimizer, state: BaselineState, grad_norm: float) -> None:
    delta = grad_norm - state.last_grad
    for group in optimizer.param_groups:
        lr = group.get("lr", 3e-4)
        group["lr"] = max(1e-6, lr - state.hyper_beta * delta)
    state.last_grad = grad_norm


def apply_pbtlite(optimizer: torch.optim.Optimizer, state: BaselineState, loss: float) -> None:
    if loss < state.best_loss:
        state.best_loss = loss
    elif loss > state.best_loss * 1.05:
        for group in optimizer.param_groups:
            group["lr"] = max(group.get("lr", 3e-4) * 0.5, 1e-6)


def apply_zclip_only(state_dict: Dict[str, Any]) -> None:
    state_dict["clip_policy"] = "ZClip"


def apply_spam_lite(state_dict: Dict[str, Any]) -> None:
    state_dict["clip_policy"] = "ZClip"
    state_dict["momentum_reset"] = True

