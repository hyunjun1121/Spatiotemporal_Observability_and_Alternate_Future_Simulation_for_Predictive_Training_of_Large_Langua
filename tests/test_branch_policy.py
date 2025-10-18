from __future__ import annotations

import json

from src.branch.bandit import UCB1, reward_from_pred
from src.branch.orchestrator import score_candidates


def test_score_candidates_ordering() -> None:
    preds = [
        {"loss_at_N": 2.0, "risk_quantile": 0.5, "recovery_time": 1.0},
        {"loss_at_N": 1.0, "risk_quantile": 0.2, "recovery_time": 0.5},
    ]
    policy = {"score": {"alpha": 1.0, "beta": 0.7, "gamma": 0.3}}
    scores = score_candidates(preds, policy)
    assert scores[1] > scores[0]


def test_ucb1_selection() -> None:
    arms = [json.dumps({"lr_delta": v}) for v in ["-10%", "0%", "+10%"]]
    bandit = UCB1(arms)
    # 초기엔 방문하지 않은 arm 선택
    first = bandit.select_arm(1)
    bandit.update(first, 1.0)
    second = bandit.select_arm(2)
    assert second in arms
    reward = reward_from_pred({"loss_at_N": 0.5, "risk_quantile": 0.1, "recovery_time": 0.2}, {"alpha": 1.0, "beta": 0.7, "gamma": 0.3})
    bandit.update(second, reward)

