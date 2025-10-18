from __future__ import annotations
from typing import Any, Dict
import torch
import random
import numpy as np


def _rng_states() -> Dict[str, Any]:
    try:
        import torch as _torch  # noqa
        cuda_state = _torch.cuda.get_rng_state_all() if _torch.cuda.is_available() else None
    except Exception:
        cuda_state = None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
    }


def gather_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    summaries: Dict[str, Any],
    epoch: int,
    global_step: int,
) -> Dict[str, Any]:
    """state_schema_v1.yaml에 맞춘 최소 상태 dict를 구성한다."""
    st: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state": getattr(scheduler, "state_dict", lambda: {})(),
        "dataloader_rng_state": None,
        "python_rng": None,
        "numpy_rng": None,
        "torch_rng": None,
        "cuda_rng": None,
        "sampler_indices_offset": 0,
        "epoch": epoch,
        "global_step": global_step,
        "summaries": summaries,
    }
    rs = _rng_states()
    st.update({
        "python_rng": rs["python"],
        "numpy_rng": rs["numpy"],
        "torch_rng": rs["torch"],
        "cuda_rng": rs["cuda"],
    })
    return st


def restore_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, state: Dict[str, Any]) -> None:
    """저장된 상태를 모델/옵티마이저/스케줄러에 복원한다."""
    model.load_state_dict(state["model_state_dict"])  # type: ignore
    optimizer.load_state_dict(state["optimizer_state_dict"])  # type: ignore
    if hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state.get("lr_scheduler_state", {}))  # type: ignore

