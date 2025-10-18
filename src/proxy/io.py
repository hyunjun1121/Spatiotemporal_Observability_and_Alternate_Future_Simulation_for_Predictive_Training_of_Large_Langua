from __future__ import annotations

import os
import torch


def save_proxy(model: torch.nn.Module, path: str) -> None:
    """Proxy 모델 체크포인트를 저장한다."""

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_proxy(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Proxy 모델 체크포인트를 로드한다."""

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model

