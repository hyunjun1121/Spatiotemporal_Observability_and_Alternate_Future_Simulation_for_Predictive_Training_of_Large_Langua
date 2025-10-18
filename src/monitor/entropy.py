from __future__ import annotations

from typing import List

import torch


def attention_entropy(attn_probs: List[torch.Tensor]) -> float:
    """per-head softmax 확률에서 평균 entropy를 계산한다 (subsample_rate 설정으로 호출 주기를 제어)."""

    entropies = []
    for weights in attn_probs:  # (B, H, T, T)
        if weights is None:
            continue
        probs = weights.clamp_min(1e-9)
        ent = -(probs * probs.log()).sum(dim=-1).mean()
        entropies.append(ent)
    if not entropies:
        return 0.0
    return float(torch.stack(entropies).mean().item())


def activation_entropy(hidden: torch.Tensor) -> float:
    """hidden state의 분산으로 Gaussian entropy 근사를 계산한다."""

    if hidden is None:
        return 0.0
    reshaped = hidden.float().reshape(-1, hidden.shape[-1])
    variance = reshaped.var(dim=0, unbiased=False) + 1e-9
    entropy = 0.5 * torch.log(2 * torch.pi * torch.e * variance)
    return float(entropy.mean().item())

