from __future__ import annotations

from typing import Dict, Any

import torch


def _scale_lr(optimizer: torch.optim.Optimizer, delta: str) -> None:
    if delta.endswith("%"):
        v = float(delta.strip("%")) / 100.0
        for g in optimizer.param_groups:
            g["lr"] = max(1e-8, g["lr"] * (1.0 + v))


def _scale_wd(optimizer: torch.optim.Optimizer, scale: float) -> None:
    for g in optimizer.param_groups:
        g["weight_decay"] = max(0.0, g.get("weight_decay", 0.0) * scale)


def _reset_momentum(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key].zero_()


def apply_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """액션을 optimizer에 적용하고 적용된 매개변수를 반환한다."""
    action = state["action"]
    optimizer: torch.optim.Optimizer = state["optimizer"]

    _scale_lr(optimizer, action.get("lr_delta", "0%"))
    _scale_wd(optimizer, float(action.get("weight_decay_scale", 1.0)))
    reset_done = False
    if int(action.get("momentum_reset", 0)) == 1:
        _reset_momentum(optimizer)
        reset_done = True
    return {
        "data_reorder": action.get("data_reorder", "none"),
        "clip_policy": action.get("clip_policy", "None"),
        "momentum_reset": reset_done,
        "lr_delta": action.get("lr_delta", "0%"),
        "weight_decay_scale": action.get("weight_decay_scale", 1.0),
    }

