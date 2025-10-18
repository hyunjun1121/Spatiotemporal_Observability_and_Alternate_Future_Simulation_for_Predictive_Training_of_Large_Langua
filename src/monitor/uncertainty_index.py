from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_uncertainty(window: Dict[str, List[float]], cfg: Dict) -> tuple[float, Dict[str, float]]:
    """Compute composite uncertainty score from recent time-series statistics."""

    eps = cfg.get("zscore_eps", 1e-6)

    def z(series: List[float]) -> float:
        arr = np.asarray(series, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        mu = float(arr.mean())
        sd = float(arr.std())
        return (float(arr[-1]) - mu) / (sd + eps)

    loss_z = z(window.get("loss", []))
    grad_series = window.get("grad_norm", [])
    grad_delta = np.diff(grad_series).tolist() if len(grad_series) >= 2 else [0.0]
    grad_z = z(grad_delta)
    volatility = 0.5 * loss_z + 0.5 * grad_z

    attn = z(window.get("attn_entropy_mean", []))
    act = z(window.get("act_entropy_mean", []))
    entropy = 0.5 * attn + 0.5 * act

    disagreement = z(window.get("forecaster_var", []))

    weights = cfg.get("weights", {"volatility": 0.5, "entropy": 0.3, "disagreement": 0.2})
    score = (
        weights.get("volatility", 0.5) * volatility
        + weights.get("entropy", 0.3) * entropy
        + weights.get("disagreement", 0.2) * disagreement
    )

    return float(score), {
        "volatility": float(volatility),
        "entropy": float(entropy),
        "disagreement": float(disagreement),
    }


def check_trigger(index: float, cfg: Dict, state: Dict[str, float | int]) -> bool:
    """Apply hysteresis and cooldown logic to determine whether to trigger branching."""

    threshold_cfg = cfg.get("threshold", {})
    trigger_z = float(threshold_cfg.get("trigger_z", 2.5))
    hysteresis_z = float(threshold_cfg.get("hysteresis_z", 3.2))
    consecutive = int(threshold_cfg.get("consecutive", 3))

    cooldown = int(state.get("cooldown_steps", cfg.get("cooldown_steps", 0)))
    current_step = int(state.get("step", 0))
    last_step = int(state.get("last_trigger_step", -10**9))
    consec_count = int(state.get("consecutive_count", 0))

    if cooldown > 0 and (current_step - last_step) < cooldown:
        state["consecutive_count"] = 0
        return False

    triggered = False
    if index >= hysteresis_z:
        triggered = True
    elif index >= trigger_z:
        consec_count += 1
        if consec_count >= consecutive:
            triggered = True
    else:
        consec_count = 0

    if triggered:
        state["last_trigger_step"] = current_step
        consec_count = 0

    state["consecutive_count"] = consec_count
    return triggered

