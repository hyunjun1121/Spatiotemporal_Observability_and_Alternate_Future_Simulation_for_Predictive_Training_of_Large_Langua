from __future__ import annotations

import json
import math
import os
from typing import Dict, List

__all__ = ["UCB1", "reward_from_pred"]


class UCB1:
    """UCB1 밴딧 구현."""

    def __init__(self, arms: List[str], state_path: str | None = None) -> None:
        self.arms = list(arms)
        self.counts = {arm: 0 for arm in self.arms}
        self.values = {arm: 0.0 for arm in self.arms}
        self.total = 0
        self.state_path = state_path
        if self.state_path and os.path.exists(self.state_path):
            self._load()

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.counts:
            self.counts[arm] = 0
            self.values[arm] = 0.0
            self.arms.append(arm)
        self.total += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        current = self.values[arm]
        self.values[arm] = current + (reward - current) / n

    def select_arm(self, _: int) -> str:
        for arm, count in self.counts.items():
            if count == 0:
                return arm
        scores: Dict[str, float] = {}
        for arm in self.arms:
            avg = self.values[arm]
            bonus = math.sqrt((2.0 * math.log(max(1, self.total))) / self.counts[arm])
            scores[arm] = avg + bonus
        return max(scores, key=scores.get)

    def save(self) -> None:
        if not self.state_path:
            return
        directory = os.path.dirname(self.state_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {"counts": self.counts, "values": self.values, "total": self.total}
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        if not self.state_path:
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self.counts.update(payload.get("counts", {}))
        self.values.update(payload.get("values", {}))
        self.total = int(payload.get("total", 0))


def reward_from_pred(pred: Dict[str, float], coeff: Dict[str, float]) -> float:
    """Proxy 예측을 바탕으로 bandit 보상을 계산한다."""

    alpha = coeff.get("alpha", 1.0)
    beta = coeff.get("beta", 0.7)
    gamma = coeff.get("gamma", 0.3)
    return -(
        alpha * pred.get("loss_at_N", 0.0)
        + beta * pred.get("risk_quantile", 0.0)
        + gamma * pred.get("recovery_time", 0.0)
    )

