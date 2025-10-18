from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from .dataset import build_proxy_dataset
from .io import save_proxy
from src.proxy.model import ProxyMLP


@dataclass
class ProxyTrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 3
    horizon: int = 2000
    window: int = 512


class _ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def _ece(pred: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(pred)
    acc = 0.0
    for i in range(bins):
        mask = (pred >= edges[i]) & (pred < edges[i + 1])
        if not mask.any():
            continue
        conf = pred[mask].mean()
        emp = target[mask].mean()
        acc += abs(conf - emp) * (mask.sum() / total)
    return float(acc)


def train_offline(log_path: str, config: Optional[ProxyTrainConfig] = None, save_path: Optional[str] = None) -> Dict[str, float]:
    cfg = config or ProxyTrainConfig()
    X, Y = build_proxy_dataset(log_path, window=cfg.window, horizon=cfg.horizon)
    if X.size == 0:
        return {"mae_loss": float("nan"), "ece": float("nan"), "samples": 0}

    dataset = _ArrayDataset(X, Y)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    model = ProxyMLP(in_dim=X.shape[1], hidden=256, layers=4)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss = float("inf")
    patience = cfg.patience
    ce = nn.L1Loss()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = ce(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                total += ce(pred, yb).item() * xb.size(0)
        val_loss = total / max(1, len(val_ds))
        if val_loss < best_loss:
            best_loss = val_loss
            patience = cfg.patience
            if save_path:
                save_proxy(model, save_path)
        else:
            patience -= 1
            if patience == 0:
                break

    return evaluate_proxy(model, val_ds)


def evaluate_proxy(model: nn.Module, dataset: Dataset) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=256)
    mae_loss = 0.0
    mae_risk = 0.0
    mae_recovery = 0.0
    preds = []
    targets = []
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            mae_loss += torch.mean(torch.abs(pred[:, 0] - yb[:, 0])).item() * xb.size(0)
            mae_risk += torch.mean(torch.abs(pred[:, 1] - yb[:, 1])).item() * xb.size(0)
            mae_recovery += torch.mean(torch.abs(pred[:, 2] - yb[:, 2])).item() * xb.size(0)
            preds.append(pred[:, 1].numpy())
            targets.append(yb[:, 1].numpy())
            total += xb.size(0)
    preds_arr = np.concatenate(preds)
    targets_arr = np.concatenate(targets)
    ece = _ece(preds_arr, targets_arr)
    return {
        "mae_loss": mae_loss / max(1, total),
        "mae_risk": mae_risk / max(1, total),
        "mae_recovery": mae_recovery / max(1, total),
        "ece": ece,
        "samples": total,
    }

