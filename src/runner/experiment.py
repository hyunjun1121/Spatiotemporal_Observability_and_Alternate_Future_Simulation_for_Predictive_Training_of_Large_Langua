from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import torch

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass
class RunContext:
    """실험 실행 시 필요한 메타데이터."""

    experiment: str
    run_id: str
    run_dir: str
    seed: int
    baseline: str
    method: str
    real_data: str
    hf_name: Optional[str]
    hf_config: Optional[str]
    config_path: Optional[str]
    devices: Optional[str]
    world_size: int
    max_concurrent: int
    resume_flag: bool


def _timestamp_hash() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    digest = hashlib.sha1(os.urandom(16)).hexdigest()[:8]
    return f"{stamp}-{digest}"


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:  # pragma: no cover
        return None


def new_run_dir(
    experiment_name: str,
    base_dir: Optional[str] = None,
    seed: int = 42,
    baseline: str = "none",
    method: str = "C",
    real_data: str = "off",
    hf_name: Optional[str] = None,
    hf_config: Optional[str] = None,
    config_snapshots: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    devices: Optional[str] = None,
    world_size: int = 1,
    max_concurrent: int = 1,
    resume_flag: bool = False,
) -> RunContext:
    """실험 run 디렉터리를 생성하고 manifest/seed/config 스냅샷을 기록한다."""

    run_id = _timestamp_hash()
    norm_baseline = baseline or "none"
    norm_method = method or "C"
    base = base_dir or os.path.join("runs", experiment_name, norm_baseline, norm_method, f"seed{seed}")
    run_dir = os.path.join(base, run_id)
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        "experiment": experiment_name,
        "run_id": run_id,
        "seed": seed,
        "baseline": norm_baseline,
        "method": norm_method,
        "real_data": real_data,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": _git_commit(),
        "world_size": world_size,
        "max_concurrent": max_concurrent,
        "resume_flag": bool(resume_flag),
    }
    if hf_name is not None:
        manifest["hf_name"] = hf_name
    if hf_config is not None:
        manifest["hf_config"] = hf_config
    if config_path is not None:
        manifest["config_path"] = config_path
    if devices is not None:
        manifest["devices"] = devices
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir, "seed_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"python": seed, "numpy": seed, "torch": seed, "torch_cuda": seed}, f, ensure_ascii=False, indent=2)

    if config_snapshots:
        if yaml is None:
            snapshot_path = os.path.join(run_dir, "config_snapshot.json")
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(config_snapshots, f, ensure_ascii=False, indent=2)
        else:
            snapshot_path = os.path.join(run_dir, "config_snapshot.yaml")
            with open(snapshot_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_snapshots, f, allow_unicode=True, sort_keys=False)
        with open(os.path.join(run_dir, "artifacts.jsonl"), "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "kind": "config",
                        "path": os.path.relpath(snapshot_path, run_dir),
                        "created_utc": datetime.utcnow().isoformat() + "Z",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return RunContext(
        experiment=experiment_name,
        run_id=run_id,
        run_dir=run_dir,
        seed=seed,
        baseline=norm_baseline,
        method=norm_method,
        real_data=real_data,
        hf_name=hf_name,
        hf_config=hf_config,
        config_path=config_path,
        devices=devices,
        world_size=world_size,
        max_concurrent=max_concurrent,
        resume_flag=bool(resume_flag),
    )


def register_artifact(run_ctx: RunContext, path: str, kind: str) -> None:
    """산출물을 artifacts.jsonl에 기록한다."""

    record = {
        "kind": kind,
        "path": os.path.relpath(path, run_ctx.run_dir),
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(run_ctx.run_dir, "artifacts.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
