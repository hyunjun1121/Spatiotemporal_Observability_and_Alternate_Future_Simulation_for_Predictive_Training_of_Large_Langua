from typing import Dict, Any
import torch, torch.nn as nn

class ProxyMLP(nn.Module):
    """Simple MLP proxy predicting [loss_at_N, risk_quantile, recovery_time]."""

    def __init__(self, in_dim: int, hidden: int, layers: int, out_dim: int = 3):
        super().__init__()
        blocks = []
        d = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(d, out_dim)  # [loss_at_N, risk_quantile, recovery_time]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

