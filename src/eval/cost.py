from __future__ import annotations

from typing import Dict


def estimate_flops(model_cfg: Dict, steps: int, seq_len: int, batch_size: int) -> float:
    hidden = model_cfg.get("hidden_size", 1024)
    layers = model_cfg.get("num_layers", 24)
    vocab = model_cfg.get("vocab_size", 50257)
    flops_per_token = 6 * layers * hidden * seq_len
    flops = flops_per_token * batch_size * steps
    return float(flops)


def estimate_energy(flops: float, hw: str = "A100-40GB") -> float:
    efficiency = 0.3 if "A100" in hw else 0.15  # pJ per FLOP
    joules = flops * efficiency * 1e-12
    return joules / 1000.0  # kJ


def estimate_co2e(energy_kj: float, grid: str = "OECD") -> float:
    factor = 0.0004 if grid == "OECD" else 0.0007  # tCO2e per kWh
    kwh = energy_kj / 3600.0
    return kwh * factor
