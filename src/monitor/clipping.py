from __future__ import annotations
from typing import Dict, Tuple
import torch


def apply_zclip(optimizer: torch.optim.Optimizer, grad_norm: float, ema_stats: Dict[str, float], z_thresh: float) -> Tuple[float, float]:
    """ZClip 근사: bound=mu+z*std를 초과하면 그 비율로 grad 스케일링.
    반환: (scale_factor, bound)
    """
    mu = float(ema_stats.get("mu", 0.0))
    sd = float(ema_stats.get("sd", 1.0))
    bound = mu + z_thresh * max(sd, 1e-6)
    scale = 1.0
    if grad_norm > bound and grad_norm > 0:
        scale = bound / grad_norm
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.detach().mul_(scale)
    return scale, bound


def apply_arc(optimizer: torch.optim.Optimizer, grad_norm: float, clip_value: float = 1.0) -> Tuple[float, float]:
    """ARC 근사: 고정 bound로 clip.
    반환: (scale_factor, bound)
    """
    bound = clip_value
    scale = 1.0
    if grad_norm > bound and grad_norm > 0:
        scale = bound / grad_norm
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.detach().mul_(scale)
    return scale, bound

