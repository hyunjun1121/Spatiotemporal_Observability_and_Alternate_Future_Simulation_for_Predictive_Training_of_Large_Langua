from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.proxy.train import evaluate_proxy
from src.proxy.model import ProxyMLP


class _DummyDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def test_proxy_calibration_basic() -> None:
    X = torch.randn(32, 8)
    Y = torch.rand(32, 3)
    dataset = _DummyDataset(X, Y)
    model = ProxyMLP(in_dim=8, hidden=32, layers=2)
    metrics = evaluate_proxy(model, dataset)
    assert 0.0 <= metrics["ece"] <= 1.0
