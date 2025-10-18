from __future__ import annotations

from src.monitor.uncertainty_index import compute_uncertainty, check_trigger


def test_uncertainty_hysteresis_and_cooldown() -> None:
    window = {
        "loss": [0.1] * 9 + [5.0],
        "grad_norm": [1.0 for _ in range(10)],
        "attn_entropy_mean": [0.1 for _ in range(10)],
        "act_entropy_mean": [0.1 for _ in range(10)],
        "forecaster_var": [0.05 for _ in range(10)],
    }
    cfg = {
        "threshold": {"trigger_z": 0.5, "hysteresis_z": 1.0, "consecutive": 2},
        "cooldown_steps": 5,
    }
    state = {"last_trigger_step": -10, "consecutive_count": 0, "cooldown_steps": 5, "step": 0}
    ui, _ = compute_uncertainty(window, cfg)
    assert check_trigger(ui, cfg, state) is False
    state["step"] = 1
    assert check_trigger(ui, cfg, state) is True
    state["step"] = 2
    assert check_trigger(ui, cfg, state) is False
