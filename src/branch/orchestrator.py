from __future__ import annotations

import json
import os
import random
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .bandit import UCB1, reward_from_pred


def _keys(space: Dict[str, Any]) -> List[str]:
    return [k for k in space.keys() if k not in ("sample_size", "topk_verify")]


def enumerate_actions(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """action space의 가능한 조합을 모두 생성한다."""

    keys = _keys(space)
    if not keys:
        return [{}]
    combos: List[Dict[str, Any]] = [{}]
    for key in keys:
        next_combos: List[Dict[str, Any]] = []
        for base in combos:
            for value in space[key]:
                candidate = dict(base)
                candidate[key] = value
                next_combos.append(candidate)
        combos = next_combos
    return combos


def sample_actions(
    space: Dict[str, List[Any]],
    count: int,
    *,
    bandit: UCB1 | None = None,
    epsilon: float = 0.1,
    t_offset: int = 0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """UCB1 + epsilon 탐색으로 action을 샘플링한다."""

    keys = _keys(space)
    actions: List[Dict[str, Any]] = []
    arm_ids: List[str] = []

    if bandit is None:
        for _ in range(count):
            action = {k: random.choice(space[k]) for k in keys}
            actions.append(action)
            arm_ids.append(json.dumps(action, sort_keys=True))
        return actions, arm_ids

    all_actions = enumerate_actions(space)
    arm_map = {json.dumps(action, sort_keys=True): action for action in all_actions}
    arm_list = list(arm_map.keys())
    for i in range(count):
        if random.random() < epsilon:
            arm_id = random.choice(arm_list)
        else:
            arm_id = bandit.select_arm(t_offset + i + 1)
        actions.append(arm_map[arm_id])
        arm_ids.append(arm_id)
    return actions, arm_ids


def score_candidates(preds: List[Dict[str, float]], policy: Dict[str, Any]) -> np.ndarray:
    """정책 가중치를 활용해 점수를 계산한다."""

    a = policy["score"]["alpha"]
    b = policy["score"]["beta"]
    c = policy["score"]["gamma"]
    scores = [-(a * p["loss_at_N"] + b * p["risk_quantile"] + c * p["recovery_time"]) for p in preds]
    return np.asarray(scores, dtype=np.float64)


def _summarise_proxy(preds: List[Dict[str, float]]) -> Dict[str, float]:
    arr = np.asarray([[p["loss_at_N"], p["risk_quantile"], p["recovery_time"]] for p in preds], dtype=np.float64)
    if arr.size == 0:
        return {"loss_mean": 0.0, "risk_mean": 0.0, "recovery_mean": 0.0, "loss_var": 0.0, "risk_var": 0.0, "recovery_var": 0.0}
    return {
        "loss_mean": float(arr[:, 0].mean()),
        "risk_mean": float(arr[:, 1].mean()),
        "recovery_mean": float(arr[:, 2].mean()),
        "loss_var": float(arr[:, 0].var()),
        "risk_var": float(arr[:, 1].var()),
        "recovery_var": float(arr[:, 2].var()),
    }


def run_branch_cycle(
    t_step: int,
    cp_path: str,
    action_space: Dict[str, Any],
    predict_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, float]]],
    verify_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    policy_cfg: Dict[str, Any],
    out_dir: str,
    *,
    bandit: UCB1 | None = None,
    epsilon: float = 0.1,
) -> Dict[str, Any]:
    """분기 사이클 전체를 실행하고 결과를 기록한다."""

    sample_count = int(action_space.get("sample_size", 128))
    topk = int(action_space.get("topk_verify", 6))

    actions, arm_ids = sample_actions(action_space, sample_count, bandit=bandit, epsilon=epsilon, t_offset=t_step)
    preds = predict_fn(actions)
    scores = score_candidates(preds, policy_cfg)
    order = np.argsort(-scores)
    top_indices = order[:topk]
    top_actions = [actions[i] for i in top_indices]
    top_arm_ids = [arm_ids[i] for i in top_indices]
    top_preds = [preds[i] for i in top_indices]

    verification = verify_fn(top_actions)
    accepted_idx = next((idx for idx, v in enumerate(verification) if v.get("accepted")), None)

    proxy_summary = _summarise_proxy(top_preds)
    proxy_gap_values = [v.get("proxy_real_gap", 0.0) for v in verification if v.get("proxy_real_gap") is not None]
    mean_proxy_gap = float(np.mean(proxy_gap_values)) if proxy_gap_values else 0.0

    if bandit is not None:
        coeff = policy_cfg.get("score", {})
        for idx, metrics in enumerate(verification):
            reward = reward_from_pred(top_preds[idx], coeff)
            if metrics.get("accepted"):
                reward += 0.1
            bandit.update(top_arm_ids[idx], reward)
        bandit.save()

    decision = {
        "t_step": t_step,
        "cp_path": cp_path,
        "candidate_count": len(actions),
        "verified_count": len(verification),
        "accepted_action": top_actions[accepted_idx] if accepted_idx is not None else None,
        "preds_topk": top_preds,
        "proxy_summary": proxy_summary,
        "verification_metrics": verification,
        "proxy_real_gap": mean_proxy_gap,
    }

    step_dir = os.path.join(out_dir, str(t_step))
    os.makedirs(step_dir, exist_ok=True)
    with open(os.path.join(step_dir, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)

    return decision

