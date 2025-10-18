from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np


def _load_records(path: str) -> List[Dict[str, float]]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    if path.endswith(".parquet"):
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas required to read parquet") from exc
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported log format: {path}")


def _feature_vector(rec: Dict[str, float]) -> List[float]:
    return [
        float(rec.get("loss", 0.0)),
        float(rec.get("grad_norm", 0.0)),
        float(rec.get("lr", 0.0)),
        float(rec.get("weight_decay", 0.0)),
        float(rec.get("clip_bound", 0.0)),
        float(1.0 if rec.get("momentum_reset", False) else 0.0),
        float(rec.get("attn_entropy_mean", 0.0)),
        float(rec.get("act_entropy_mean", 0.0)),
    ]


def _target_vector(rec: Dict[str, float]) -> List[float]:
    loss_target = float(rec.get("loss", 0.0))
    risk = float(rec.get("forecaster_var", 0.0)) ** 0.5
    risk = max(0.0, min(1.0, risk))
    recovery = float(rec.get("branch", {}).get("verified", 0))
    return [loss_target, risk, recovery]


def build_proxy_dataset(log_path: str, window: int = 512, horizon: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Proxy 학습용 rolling window dataset을 구축한다."""
    records = _load_records(log_path)
    if not records:
        return np.empty((0, 8), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    X: List[List[float]] = []
    Y: List[List[float]] = []
    for idx in range(len(records)):
        if idx < window:
            continue
        future_idx = min(len(records) - 1, idx + horizon)
        feat = _feature_vector(records[idx])
        targ = _target_vector(records[future_idx])
        X.append(feat)
        Y.append(targ)

    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)

