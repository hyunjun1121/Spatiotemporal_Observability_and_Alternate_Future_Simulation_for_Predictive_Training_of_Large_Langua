from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np


def _volatility(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanstd(arr))


def run_verification(
    step_fn: Callable[[], Dict[str, float]],
    steps: int,
    baseline: Dict[str, float],
    history_window: List[float],
    dataset_mode: str = "synthetic",
    expected_loss: float | None = None,
) -> Dict[str, Any]:
    """branch 후보에 대한 짧은 롤아웃을 수행해 수용 여부를 판단한다."""

    history: List[Dict[str, float]] = []
    losses: List[float] = []
    surrogate_vals: List[float] = []

    for _ in range(steps):
        metrics = step_fn()
        history.append(metrics)
        losses.append(metrics.get("loss", np.nan))
        surrogate_vals.append(metrics.get("val_loss", metrics.get("loss", np.nan)))

    recent_vol = _volatility(history_window[-200:]) if history_window else float("inf")
    new_vol = _volatility(losses)
    volatility_drop = np.isfinite(new_vol) and new_vol < recent_vol

    val_loss_avg = float(np.nanmean(surrogate_vals))
    val_loss_delta = float(baseline.get("val_loss", float("inf")) - val_loss_avg)
    if dataset_mode != "synthetic":
        val_improved = val_loss_delta > 0.0
    else:
        val_improved = val_loss_delta > 0.0

    last_loss = float(history[-1].get("loss", val_loss_avg)) if history else val_loss_avg
    proxy_gap = abs(last_loss - expected_loss) if expected_loss is not None else 0.0

    accepted = volatility_drop or val_improved

    return {
        "accepted": bool(accepted),
        "volatility_drop": bool(volatility_drop),
        "val_improved": bool(val_improved),
        "val_loss_delta": val_loss_delta,
        "proxy_real_gap": float(proxy_gap),
        "history": history,
    }

