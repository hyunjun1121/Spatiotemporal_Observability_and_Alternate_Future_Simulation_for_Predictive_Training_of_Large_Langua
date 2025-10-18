from __future__ import annotations

import json
import math
import os
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.branch.actions import apply_action
from src.branch.bandit import UCB1
from src.branch.orchestrator import enumerate_actions, run_branch_cycle
from src.branch.verification import run_verification
from src.cp.checkpoint import save_state
from src.cp.state import gather_state
from src.data.datasets import build_dataloader
from src.monitor.clipping import apply_arc, apply_zclip
from src.monitor.entropy import attention_entropy, activation_entropy
from src.monitor.forecaster import LossForecasterEnsemble
from src.monitor.uncertainty_index import compute_uncertainty, check_trigger
from src.proxy.io import load_proxy
from src.proxy.model import ProxyMLP
from src.runner.experiment import RunContext, new_run_dir, register_artifact
from src.train.baselines import (
    BaselineState,
    apply_fixed_cosine,
    apply_hypergradient,
    apply_pbtlite,
    apply_spam_lite,
    apply_zclip_only,
)
from src.train.build import build_model, build_optimizer, build_scheduler
from src.utils.config import load_yaml, validate_log_record
from src.utils.io import append_parquet, ensure_dir, write_jsonl
from src.utils.seed import set_all_seeds
from src.eval.aggregate import aggregate_run


def _init_ddp() -> Dict[str, int]:
    """환경 변수에 따라 DDP 초기화를 수행한다."""

    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        return {"rank": dist.get_rank(), "world": dist.get_world_size()}
    return {"rank": 0, "world": 1}


def _device_for_rank(rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("cpu")


def _select_precision(requested: str) -> str:
    req = requested.lower()
    if req in {"bf16", "fp16"} and not torch.cuda.is_available():
        warnings.warn("CUDA is unavailable; falling back to fp32 precision.")
        return "fp32"
    return requested


def _feature_vector(record: Dict[str, float]) -> List[float]:
    return [
        float(record.get("loss", 0.0)),
        float(record.get("grad_norm", 0.0)),
        float(record.get("lr", 0.0)),
        float(record.get("weight_decay", 0.0)),
        float(record.get("clip_bound", 0.0)),
        float(1.0 if record.get("clip_policy") == "ZClip" else 0.0),
        float(record.get("attn_entropy_mean", 0.0)),
        float(record.get("act_entropy_mean", 0.0)),
    ]


def _broadcast_run_context(
    run_ctx: Optional[RunContext], *, experiment: str, seed: int, world: int, rank: int
) -> RunContext:
    if world == 1:
        assert run_ctx is not None
        return run_ctx
    payload = [
        run_ctx.run_dir if run_ctx else None,
        run_ctx.run_id if run_ctx else None,
        run_ctx.baseline if run_ctx else None,
        run_ctx.method if run_ctx else None,
        run_ctx.real_data if run_ctx else None,
        run_ctx.config_path if run_ctx else None,
        run_ctx.devices if run_ctx else None,
        run_ctx.world_size if run_ctx else 1,
        run_ctx.max_concurrent if run_ctx else 1,
        run_ctx.resume_flag if run_ctx else False,
    ]
    dist.broadcast_object_list(payload, src=0)
    if rank == 0:
        return run_ctx  # type: ignore[arg-type]
    return RunContext(
        experiment=experiment,
        run_id=payload[1],
        run_dir=payload[0],
        seed=seed,
        baseline=payload[2] or "none",
        method=payload[3] or "C",
        real_data=payload[4] or "off",
        config_path=payload[5],
        devices=payload[6],
        world_size=int(payload[7] or 1),
        max_concurrent=int(payload[8] or 1),
        resume_flag=bool(payload[9]),
    )


def train_baseline(
    dataset_mode: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    seed: int = 42,
    precision: str = "bf16",
    experiment_name: str = "baseline",
    run_dir: Optional[str] = None,
    cooldown: int = 300,
    epsilon_ucb: float = 0.1,
    ablation_mode: str = "C",
    real_data: str = "off",
    hf_name: Optional[str] = None,
    offline_data_dir: Optional[str] = None,
    baseline: str = "none",
    method: str = "C",
    replicate_id: int = 0,
    config_path: Optional[str] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
    devices: Optional[str] = None,
    world_size: int = 1,
    max_concurrent: int = 1,
    resume_flag: bool = False,
) -> None:
    """Minimal Trainer + Monitor + Proxy + Branch Orchestrator 통합 루프."""

    ddp = _init_ddp()
    rank, world = ddp["rank"], ddp["world"]
    device = _device_for_rank(rank)
    is_master = rank == 0

    set_all_seeds(seed)

    unc_cfg = load_yaml("assets/uncertainty_config.yaml")
    hparam_space = load_yaml("assets/hparam_space.yaml")
    branch_policy = load_yaml("assets/branch_policy.yaml")
    proxy_cfg = load_yaml("assets/proxy_config.yaml")
    noise_cfg = load_yaml("assets/noise_config.yaml")
    state_schema = load_yaml("assets/state_schema_v1.yaml")
    log_schema_path = "assets/log_schema.json"

    policy_cooldown = int(branch_policy.get("cooldown_steps", cooldown))
    cooldown = policy_cooldown
    unc_cfg["cooldown_steps"] = cooldown

    entropy_rate = int(proxy_cfg.get("subsample_rate", 50))
    forecaster_interval = entropy_rate
    proxy_checkpoint = proxy_cfg.get("checkpoint_path", os.path.join("runs", "proxy", "proxy_mlp.pt"))

    config_snapshot = {
        "uncertainty_config": unc_cfg,
        "branch_policy": branch_policy,
        "proxy_config": proxy_cfg,
        "noise_config": noise_cfg,
        "dataset_mode": dataset_mode,
        "real_data": real_data,
        "baseline": baseline,
        "method": method,
        "run": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "steps": steps,
        },
    }
    if experiment_config is not None:
        config_snapshot["experiment_config"] = experiment_config
    if config_path is not None:
        config_snapshot["experiment_config_path"] = config_path
    config_snapshot["scheduler"] = {
        "devices": devices,
        "world_size": world_size,
        "max_concurrent": max_concurrent,
        "resume_flag": resume_flag,
    }

    run_ctx: Optional[RunContext] = None
    if is_master:
        run_ctx = new_run_dir(
            experiment_name,
            base_dir=run_dir,
            seed=seed,
            baseline=baseline,
            method=method,
            real_data=real_data,
            config_snapshots=config_snapshot,
            config_path=config_path,
            devices=devices,
            world_size=world_size,
            max_concurrent=max_concurrent,
            resume_flag=resume_flag,
        )
    run_ctx = _broadcast_run_context(run_ctx, experiment=experiment_name, seed=seed, world=world, rank=rank)

    loader = build_dataloader(
        dataset_mode,
        noise_cfg,
        batch_size,
        seq_len,
        real_data=real_data,
        hf_name=hf_name,
        offline_data_dir=offline_data_dir,
    )

    effective_precision = _select_precision(precision)
    model = build_model(state_schema.get("model", {}).get("arch", "gpt2-medium"), precision=effective_precision)
    model.to(device)
    if world > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = build_optimizer(model, weight_decay=0.01, lr=3e-4)
    scheduler = build_scheduler(optimizer, state_schema.get("lr_scheduler", {}), total_steps=steps)

    feature_dim = 8
    forecaster = LossForecasterEnsemble(in_dim=feature_dim, K=3, device=str(device), hidden=64, quantile=0.8)

    proxy_model: Optional[ProxyMLP] = None
    if os.path.exists(proxy_checkpoint):
        proxy_model = ProxyMLP(in_dim=feature_dim, hidden=proxy_cfg.get("hidden_dim", 256), layers=proxy_cfg.get("layers", 4))
        load_proxy(proxy_model, proxy_checkpoint)
        proxy_model.eval()

    arms = [json.dumps(action, sort_keys=True) for action in enumerate_actions(hparam_space)]
    bandit: Optional[UCB1] = None
    if arms:
        bandit_state_path = os.path.join(run_ctx.run_dir, "bandit_state.json")
        bandit = UCB1(arms, state_path=bandit_state_path)

    ensure_dir(run_ctx.run_dir)
    log_jsonl = os.path.join(run_ctx.run_dir, "train_log.jsonl")
    log_parquet = os.path.join(run_ctx.run_dir, "log_db.parquet")

    parquet_buffer: List[Dict[str, Any]] = []
    feature_window: List[List[float]] = []
    loss_history: List[float] = []
    uncertainty_window: Dict[str, List[float]] = {
        "loss": [],
        "grad_norm": [],
        "attn_entropy_mean": [],
        "act_entropy_mean": [],
        "forecaster_var": [],
    }
    trigger_state: Dict[str, Any] = {"last_trigger_step": -10**9, "consecutive_count": 0, "cooldown_steps": cooldown}

    ema_mu = 0.0
    ema_sq = 0.0
    ema_beta = float(unc_cfg.get("ema_decay", 0.98))

    wasted_flops_est = 0.0
    branch_calls = 0
    accept_count = 0

    state: Dict[str, Any] = {"clip_policy": "None", "last_action_applied": {}, "baseline": baseline}
    baseline_state = BaselineState(method=baseline)
    prediction_cache: Dict[str, Dict[str, float]] = {}

    iterator = iter(loader)

    def _forward_step(inputs: torch.Tensor, targets: torch.Tensor, *, collect_attn: bool) -> Dict[str, float]:
        nonlocal ema_mu, ema_sq

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(inputs.to(device), return_attentions=collect_attn)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.to(device).reshape(-1))
        loss.backward()

        grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_sq += param.grad.data.norm(2).item() ** 2
        grad_norm = math.sqrt(max(grad_norm_sq, 0.0))

        ema_mu = ema_beta * ema_mu + (1 - ema_beta) * grad_norm
        ema_sq = ema_beta * ema_sq + (1 - ema_beta) * (grad_norm ** 2)
        sd = max(1e-6, math.sqrt(max(0.0, ema_sq - ema_mu * ema_mu)))
        zscore = (grad_norm - ema_mu) / sd if sd > 0 else 0.0

        clip_policy = state.get("clip_policy", "None")
        clip_bound = 0.0
        if clip_policy == "ZClip":
            _, clip_bound = apply_zclip(optimizer, grad_norm, {"mu": ema_mu, "sd": sd}, z_thresh=2.5)
        elif clip_policy == "ARC":
            _, clip_bound = apply_arc(optimizer, grad_norm, clip_value=1.0)

        optimizer.step()
        scheduler.step()

        attn_entropy_val = 0.0
        act_entropy_val = 0.0
        if collect_attn:
            attn_entropy_val = attention_entropy(aux.get("attentions", []))
            act_entropy_val = activation_entropy(aux.get("hidden_last"))

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "grad_zscore": float(zscore),
            "clip_policy": clip_policy,
            "clip_bound": float(clip_bound),
            "attn_entropy_mean": float(attn_entropy_val),
            "act_entropy_mean": float(act_entropy_val),
        }

    def _predict_single(fallback: Dict[str, float]) -> Dict[str, float]:
        if proxy_model is not None and ablation_mode != "A":
            if feature_window:
                inputs = torch.tensor(feature_window[-1], dtype=torch.float32).unsqueeze(0)
            else:
                inputs = torch.zeros((1, feature_dim), dtype=torch.float32)
            with torch.no_grad():
                out = proxy_model(inputs).squeeze(0).tolist()
            return {
                "loss_at_N": float(out[0]),
                "risk_quantile": float(out[1]),
                "recovery_time": float(out[2]),
            }
        return fallback

    for step in range(steps):
        inputs, targets = next(iterator)
        collect_attn = (step % max(1, entropy_rate)) == 0
        metrics = _forward_step(inputs, targets, collect_attn=collect_attn)
        state["clip_policy"] = metrics.get("clip_policy", "None")

        record = {
            "loss": metrics["loss"],
            "grad_norm": metrics["grad_norm"],
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
            "clip_bound": metrics.get("clip_bound", 0.0),
            "clip_policy": metrics.get("clip_policy", "None"),
            "attn_entropy_mean": metrics.get("attn_entropy_mean", 0.0),
            "act_entropy_mean": metrics.get("act_entropy_mean", 0.0),
        }

        feature = _feature_vector(record)
        feature_window.append(feature)
        if len(feature_window) > 512:
            feature_window.pop(0)

        if baseline == "fixedlr":
            apply_fixed_cosine(scheduler)
        elif baseline == "hypergrad":
            apply_hypergradient(optimizer, baseline_state, metrics["grad_norm"])
        elif baseline == "pbtlite":
            apply_pbtlite(optimizer, baseline_state, metrics["loss"])
        elif baseline == "zclip":
            apply_zclip_only(state)
        elif baseline == "spamlite":
            apply_spam_lite(state)

        loss_history.append(metrics["loss"])
        if len(loss_history) > 1024:
            loss_history.pop(0)

        if step % max(1, forecaster_interval) == 0 and feature_window:
            window_tensor = np.asarray(feature_window[-50:], dtype=np.float32)[np.newaxis, ...]
            forecaster.fit_online(window_tensor, metrics["loss"])

        predict_input = (
            np.asarray(feature_window[-50:], dtype=np.float32)[np.newaxis, ...]
            if feature_window
            else np.zeros((1, 1, feature_dim), dtype=np.float32)
        )
        fore_pred = forecaster.predict(predict_input, horizon=2000)
        proxy_output = {
            "loss_at_N": float(fore_pred.get("mean", metrics["loss"])),
            "risk_quantile": float(fore_pred.get("risk_quantile", 0.0)),
            "recovery_time": float(fore_pred.get("var", 0.0)),
        }

        uncertainty_window["loss"].append(metrics["loss"])
        uncertainty_window["grad_norm"].append(metrics["grad_norm"])
        uncertainty_window["attn_entropy_mean"].append(metrics.get("attn_entropy_mean", 0.0))
        uncertainty_window["act_entropy_mean"].append(metrics.get("act_entropy_mean", 0.0))
        uncertainty_window["forecaster_var"].append(fore_pred.get("var", 0.0))
        for key in uncertainty_window:
            if len(uncertainty_window[key]) > 512:
                uncertainty_window[key].pop(0)

        if ablation_mode == "D":
            length = len(uncertainty_window["loss"])
            filtered = {
                "loss": uncertainty_window["loss"],
                "grad_norm": uncertainty_window["grad_norm"],
                "attn_entropy_mean": [0.0] * length,
                "act_entropy_mean": [0.0] * length,
                "forecaster_var": [0.0] * length,
            }
            uncertainty, _ = compute_uncertainty(filtered, unc_cfg)
        else:
            uncertainty, _ = compute_uncertainty(uncertainty_window, unc_cfg)

        last_trigger_before = trigger_state.get("last_trigger_step", -10**9)
        cooldown_active = (step - last_trigger_before) < cooldown
        trigger_state["step"] = step
        triggered = check_trigger(uncertainty, unc_cfg, trigger_state)
        if triggered:
            cooldown_active = False

        decision: Dict[str, Any] = {}
        current_gap = 0.0
        prediction_cache.clear()

        def _predict_fn(actions: List[Dict[str, Any]]) -> List[Dict[str, float]]:
            preds: List[Dict[str, float]] = []
            for action in actions:
                pred = _predict_single(proxy_output)
                key = json.dumps(action, sort_keys=True)
                prediction_cache[key] = pred
                preds.append(pred)
            return preds

        def _verify_fn(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if ablation_mode == "B":
                return [
                    {
                        "accepted": True,
                        "volatility_drop": False,
                        "val_improved": False,
                        "val_loss_delta": 0.0,
                        "proxy_real_gap": 0.0,
                        "history": [],
                        "wasted": 0.0,
                    }
                    for _ in actions
                ]

            base = {
                "volatility": float(np.std(loss_history[-200:])) if loss_history else float("inf"),
                "val_loss": float(np.mean(loss_history[-200:])) if loss_history else float("inf"),
            }
            results: List[Dict[str, Any]] = []
            for action in actions:
                applied = apply_action({"action": action, "optimizer": optimizer})
                state["clip_policy"] = applied.get("clip_policy", "None")
                state["last_action_applied"] = applied
                expected = prediction_cache.get(json.dumps(action, sort_keys=True), proxy_output)
                verification = run_verification(
                    lambda: _forward_step(inputs, targets, collect_attn=False),
                    steps=50,
                    baseline=base,
                    history_window=loss_history,
                    dataset_mode=dataset_mode,
                    expected_loss=expected.get("loss_at_N"),
                )
                verification["wasted"] = 50 * batch_size
                results.append(verification)
            return results

        if triggered and is_master:
            branch_calls += 1
            branches_dir = os.path.join(run_ctx.run_dir, "branches")
            ensure_dir(branches_dir)
            cp_path = os.path.join(branches_dir, f"cp_{step:06d}.ptz")
            state_payload = gather_state(
                model,
                optimizer,
                scheduler,
                summaries={
                    "loss_ema": float(np.mean(loss_history[-100:])) if loss_history else 0.0,
                    "grad_norm_ema": float(np.mean([vec[1] for vec in feature_window[-100:]])) if feature_window else 0.0,
                    "attention_entropy_by_layer": [],
                    "activation_entropy_by_layer": [],
                },
                epoch=0,
                global_step=step,
            )
            save_state(cp_path, state_payload)
            register_artifact(run_ctx, cp_path, "checkpoint")

            decision = run_branch_cycle(
                t_step=step,
                cp_path=cp_path,
                action_space=hparam_space,
                predict_fn=_predict_fn,
                verify_fn=_verify_fn,
                policy_cfg=branch_policy,
                out_dir=branches_dir,
                bandit=bandit,
                epsilon=epsilon_ucb,
            )
            register_artifact(run_ctx, os.path.join(branches_dir, str(step), "decision.json"), "decision")

            verification_metrics = decision.get("verification_metrics", [])
            wasted_flops_est += sum(item.get("wasted", 0.0) for item in verification_metrics)
            if decision.get("accepted_action") is not None:
                accept_count += 1
            current_gap = float(decision.get("proxy_real_gap", 0.0))

        branch_info = {
            "candidate_count": decision.get("candidate_count", 0),
            "verified": decision.get("verified_count", 0),
            "accepted_action": decision.get("accepted_action"),
        }

        log_record: Dict[str, Any] = {
            "run_id": run_ctx.run_id,
            "experiment": run_ctx.experiment,
            "step": int(step),
            "walltime": float(time.time()),
            "loss": float(metrics["loss"]),
            "val_loss": float(metrics["loss"]),
            "grad_norm": float(metrics["grad_norm"]),
            "grad_zscore": float(metrics.get("grad_zscore", 0.0)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
            "clip_policy": str(metrics.get("clip_policy", "None")),
            "clip_bound": float(metrics.get("clip_bound", 0.0)),
            "momentum_reset": bool(state.get("last_action_applied", {}).get("momentum_reset", False)),
            "attn_entropy": [],
            "act_entropy": [],
            "attn_entropy_mean": float(metrics.get("attn_entropy_mean", 0.0)),
            "act_entropy_mean": float(metrics.get("act_entropy_mean", 0.0)),
            "uncertainty_index": float(uncertainty),
            "triggered": bool(triggered),
            "event_spike_flag": bool(metrics.get("grad_zscore", 0.0) >= 2.5),
            "event_trigger_flag": bool(triggered),
            "forecaster_mean": float(proxy_output["loss_at_N"]),
            "forecaster_var": float(fore_pred.get("var", 0.0)),
            "proxy_pred": proxy_output,
            "decision": {
                "action_id": json.dumps(decision.get("accepted_action")) if decision.get("accepted_action") is not None else None,
                "accepted": decision.get("accepted_action") is not None,
                "reason": "",
            },
            "branch": branch_info,
            "cooldown_active": bool(cooldown_active),
            "accept_rate": accept_count / max(1, branch_calls),
            "proxy_real_gap": float(current_gap),
            "wasted_flops_est": float(wasted_flops_est),
            "action_applied": state.get("last_action_applied", {}),
        }

        if is_master:
            try:
                validate_log_record(log_record, log_schema_path)
            except Exception:
                pass
            write_jsonl(log_jsonl, log_record)
            parquet_buffer.append(log_record)
            if len(parquet_buffer) >= 50:
                append_parquet(log_parquet, parquet_buffer)
                parquet_buffer.clear()

    if is_master:
        if parquet_buffer:
            append_parquet(log_parquet, parquet_buffer)
        try:
            aggregate_run(run_ctx.run_dir)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Failed to aggregate run metrics: {exc}")
