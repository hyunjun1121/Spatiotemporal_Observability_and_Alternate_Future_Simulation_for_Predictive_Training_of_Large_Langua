from __future__ import annotations

import io
import os
import tempfile

import torch

from src.cp.checkpoint import save_state, load_state
from src.cp.state import gather_state, restore_state
from src.train.build import build_model, build_optimizer, build_scheduler


def test_checkpoint_roundtrip() -> None:
    model = build_model()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, {"warmup_steps": 10}, total_steps=20)
    state = gather_state(model, optimizer, scheduler, summaries={}, epoch=0, global_step=5)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.ptz")
        save_state(path, state)
        loaded = load_state(path)

    model2 = build_model()
    optimizer2 = build_optimizer(model2)
    scheduler2 = build_scheduler(optimizer2, {"warmup_steps": 10}, total_steps=20)
    restore_state(model2, optimizer2, scheduler2, loaded)

    for p, q in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p, q)

