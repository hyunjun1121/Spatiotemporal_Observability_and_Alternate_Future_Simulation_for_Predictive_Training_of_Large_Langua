from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover - PyYAML 없으면 즉시 알림
    raise SystemExit(f"PyYAML is required for run orchestration: {exc}") from exc


# ---------------------------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------------------------


def _split_list(value: Optional[str], default: Iterable[str]) -> List[str]:
    if value is None:
        return list(default)
    if value == "all":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _detect_gpu_count() -> int:
    """CUDA 장치 수 추정. 실패 시 1."""

    env_count = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_count:
        return len([d for d in env_count.split(",") if d.strip()])

    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            return max(1, len(lines))
    except FileNotFoundError:
        pass
    return 1


def _preflight_offline_cache(offline_dir: Path, cache_index: Path) -> None:
    """Offline shard 경로 검사 및 캐시 인덱스 생성."""

    if not offline_dir.exists():
        raise SystemExit(f"offline_data_dir not found: {offline_dir}")
    if not offline_dir.is_dir():
        raise SystemExit(f"offline_data_dir is not a directory: {offline_dir}")

    records: List[Dict[str, object]] = []

    for file_path in sorted(offline_dir.rglob("*")):
        if not file_path.is_file():
            continue
        stat = file_path.stat()
        hasher = hashlib.md5()
        with file_path.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        records.append(
            {
                "relative_path": str(file_path.relative_to(offline_dir)),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "md5": hasher.hexdigest(),
            }
        )

    cache_index.parent.mkdir(parents=True, exist_ok=True)
    cache_index.write_text(
        json.dumps(
            {
                "offline_dir": str(offline_dir),
                "generated_utc": datetime.utcnow().isoformat() + "Z",
                "files": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _seed_dir(root: Path, experiment: str, baseline: str, method: str, seed: int) -> Path:
    return root / experiment / baseline / method / f"seed{seed}"


def _mark_done(seed_dir: Path) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / ".done").write_text(datetime.utcnow().isoformat() + "Z", encoding="utf-8")


def _seed_completed(seed_dir: Path) -> bool:
    return (seed_dir / ".done").exists()


def _write_failure(seed_dir: Path, attempt: int, code: int, stderr: str) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "attempts": attempt,
        "exit_code": code,
        "stderr": stderr[-4000:],  # 과도한 길이 방지
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (seed_dir / "run_failure.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _budget_signal_passed(signal_path: Path, experiment: str) -> bool:
    if not signal_path.exists():
        return False
    try:
        data = json.loads(signal_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    info = data.get("datasets", {}).get(experiment)
    if not isinstance(info, dict):
        return False
    return bool(info.get("passed"))


def _run_subprocess(cmd: List[str], env: Dict[str, str], cwd: Path) -> Tuple[int, str]:
    result = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    stderr = (result.stderr or "") + (result.stdout or "")
    return result.returncode, stderr


# ---------------------------------------------------------------------------
# 메인 스케줄러 로직
# ---------------------------------------------------------------------------


class Scheduler:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path.cwd()
        self.runs_root = Path(args.runs_root).resolve()
        self.paper_dir = Path(args.paper_dir).resolve()
        self.lock_json = Path(args.lock_json)
        self.signal_path = (
            self.runs_root / args.experiment_name / "budget_signal.json"
            if args.budget_signal is None
            else Path(args.budget_signal)
        )
        self.jobs: List[Tuple[str, str]] = []
        self.completed_jobs: List[Tuple[str, str]] = []
        self.stop_event = threading.Event()

    def prepare_jobs(self, baselines: Iterable[str], methods: Iterable[str]) -> None:
        combos = []
        for baseline in baselines:
            for method in methods:
                combos.append((baseline, method))
        self.jobs = combos

    # --------------------------------------------
    # 단일 조합 실행 (replicate × seed loop)
    # --------------------------------------------
    def run_combo(self, baseline: str, method: str) -> Dict[str, object]:
        seeds = self.args.seeds
        success = True
        seed_failures: List[Dict[str, object]] = []
        for seed in seeds:
            if self.stop_event.is_set():
                break
            seed_dir = _seed_dir(self.runs_root, self.args.experiment_name, baseline, method, seed)
            if self.args.resume and _seed_completed(seed_dir) and not self.args.dry_run:
                continue

            if self.args.dry_run:
                seed_dir.mkdir(parents=True, exist_ok=True)
                (seed_dir / ".dry_run").touch()
                continue

            attempts = 0
            while True:
                attempts += 1
                env = os.environ.copy()
                if self.args.devices:
                    env["CUDA_VISIBLE_DEVICES"] = self.args.devices
                env["PYTHONPATH"] = env.get("PYTHONPATH", str(self.repo_root))
                env["WORLD_SIZE"] = str(self.args.world_size)
                env["RUN_MAX_CONCURRENT"] = str(self.args.max_concurrent)
                env["RUN_RESUME_FLAG"] = "1" if self.args.resume else "0"

                seed_dir.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "-m",
                    "src.main",
                    "--mode",
                    "baseline",
                    "--dataset",
                    self.args.dataset_mode,
                    "--real_data",
                    self.args.real_data,
                    "--hf_name",
                    self.args.hf_name,
                    "--seq_len",
                    str(self.args.seq_len),
                    "--batch_size",
                    str(self.args.batch_size),
                    "--steps",
                    str(self.args.steps),
                    "--seed",
                    str(seed),
                    "--replicates",
                    "1",
                    "--baseline",
                    baseline,
                    "--method",
                    method,
                    "--experiment_name",
                    self.args.experiment_name,
                    "--run_dir",
                    str(seed_dir),
                    "--cooldown",
                    str(self.args.cooldown),
                    "--epsilon_ucb",
                    str(self.args.epsilon_ucb),
                    "--config",
                    str(self.args.config_path),
                    "--world_size",
                    str(self.args.world_size),
                    "--max_concurrent",
                    str(self.args.max_concurrent),
                ]
                if self.args.devices:
                    cmd.extend(["--devices", self.args.devices])
                if self.args.resume:
                    cmd.append("--resume_flag")

                if self.args.offline_data_dir:
                    cmd.extend(["--offline_data_dir", str(self.args.offline_data_dir)])

                code, stderr = _run_subprocess(cmd, env=env, cwd=self.repo_root)
                if code == 0:
                    _mark_done(seed_dir)
                    failure_log = seed_dir / "run_failure.json"
                    if failure_log.exists():
                        try:
                            failure_log.unlink()
                        except OSError:
                            pass
                    break

                _write_failure(seed_dir, attempts, code, stderr)
                if attempts > self.args.retries:
                    success = False
                    seed_failures.append(
                        {
                            "seed": seed,
                            "attempts": attempts,
                            "exit_code": code,
                        }
                    )
                    break

        return {
            "baseline": baseline,
            "method": method,
            "success": success,
            "failures": seed_failures,
        }

    # --------------------------------------------
    # 전체 스케줄 흐름
    # --------------------------------------------
    def execute(self) -> None:
        if not self.jobs:
            print("[scheduler] No jobs to execute.")
            return

        max_workers = max(1, self.args.max_concurrent)
        futures: List[Future] = []
        job_iter = iter(self.jobs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 초기 워커 채우기
            while len(futures) < max_workers:
                try:
                    baseline, method = next(job_iter)
                except StopIteration:
                    break
                futures.append(executor.submit(self.run_combo, baseline, method))

            while futures:
                done, pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.remove(future)
                    try:
                        result = future.result()
                        self.completed_jobs.append((result["baseline"], result["method"]))
                        if result["success"]:
                            print(f"[scheduler] combo {result['baseline']} / {result['method']} completed.")
                        else:
                            print(
                                f"[scheduler] combo {result['baseline']} / {result['method']} failed seeds: {result['failures']}"
                            )
                    except Exception as exc:  # pragma: no cover - 예상치 못한 실패
                        print(f"[scheduler] combo execution error: {exc}", file=sys.stderr)
                        self.stop_event.set()
                        continue

                    if not self.args.dry_run:
                        self._invoke_aggregate()
                        if self.args.budget_stop and _budget_signal_passed(self.signal_path, self.args.experiment_name):
                            print(
                                f"[scheduler] early stop triggered by lock criteria for {self.args.experiment_name}."
                            )
                            self.stop_event.set()
                            # 남은 future 취소
                            for pending_future in pending:
                                pending_future.cancel()
                            return

                if self.stop_event.is_set():
                    break

                # 새 작업 제출
                while not self.stop_event.is_set():
                    try:
                        baseline, method = next(job_iter)
                    except StopIteration:
                        break
                    futures.append(executor.submit(self.run_combo, baseline, method))
                    if len(futures) >= max_workers:
                        break

    def _invoke_aggregate(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "src.eval.aggregate",
            "--runs_root",
            str(self.runs_root),
            "--paper_dir",
            str(self.paper_dir),
            "--groupby",
            self.args.groupby,
            "--lock_json",
            str(self.lock_json),
            "--budget_signal",
            str(self.signal_path),
        ]
        subprocess.run(cmd, cwd=self.repo_root, check=False)


# ---------------------------------------------------------------------------
# Argparse 및 초기 설정
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replicate runner with scheduler/resource controller v1")
    parser.add_argument("--real_data", default="wikitext")
    parser.add_argument("--baseline", default="fixedlr")
    parser.add_argument("--method", default="C")
    parser.add_argument("--config", default="assets/experiments/wikitext_rc1.yaml")
    parser.add_argument("--runs_root", default="runs")
    parser.add_argument("--paper_dir", default="paper")
    parser.add_argument("--groupby", default="experiment,baseline,method")
    parser.add_argument("--lock_json", default="lock.json")
    parser.add_argument("--budget_signal", default=None)
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")

    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--world_size_auto", action="store_true")
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--budget_stop", action="store_true")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline_data_dir", default=None)

    parser.add_argument("--seeds", default="1337,2337,3337")
    parser.add_argument("--paper_only", action="store_true", help="테이블/도표 업데이트만 수행")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)

    dataset_mode = config.get("dataset", "hf")
    real_data = config.get("real_data", args.real_data)
    hf_name = config.get("hf_name", "wikitext-103")
    seq_len = int(config.get("seq_len", 1024))
    batch_size = int(config.get("batch_size", 16))
    steps = int(config.get("steps", 50000))
    replicates = int(config.get("replicates", 3))
    cooldown = int(config.get("cooldown", 300))
    epsilon_ucb = float(config.get("epsilon_ucb", 0.1))
    experiment_name = str(config.get("experiment_name", config.get("experiment", config_path.stem)))

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    if replicates and len(seeds) != replicates:
        print("[scheduler] replicates mismatch – overriding seeds based on config.")
        seeds = [1337, 2337, 3337][:replicates]

    runs_root = Path(args.runs_root).resolve()
    paper_dir = Path(args.paper_dir).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    baselines_all = ["fixedlr", "hypergrad", "zclip", "spamlite", "pbtlite"]
    methods_all = ["A", "B", "C", "D"]

    baselines = _split_list(args.baseline, baselines_all)
    methods = _split_list(args.method, methods_all)

    offline_data_dir = Path(args.offline_data_dir) if args.offline_data_dir else None
    if offline_data_dir is None and config.get("offline_data_dir"):
        offline_data_dir = Path(config["offline_data_dir"])

    if offline_data_dir:
        cache_index = runs_root / experiment_name / "offline_cache_index.json"
        print(f"[scheduler] preflight offline shards at {offline_data_dir}")
        _preflight_offline_cache(offline_data_dir, cache_index)

    if args.paper_only:
        # 통계/테이블만 갱신
        Scheduler(
            argparse.Namespace(
                runs_root=str(runs_root),
                paper_dir=str(paper_dir),
                lock_json=args.lock_json,
                budget_signal=args.budget_signal,
                groupby=args.groupby,
                experiment_name=experiment_name,
                dry_run=True,
                budget_stop=False,
                max_concurrent=1,
                world_size=1,
                devices=None,
                resume=False,
                seeds=[],
                dataset_mode=dataset_mode,
                real_data=real_data,
                hf_name=hf_name,
                seq_len=seq_len,
                batch_size=batch_size,
                steps=steps,
                cooldown=cooldown,
                epsilon_ucb=epsilon_ucb,
                config_path=config_path,
                retries=0,
                offline_data_dir=offline_data_dir,
            )
        )._invoke_aggregate()
        return

    if args.world_size_auto:
        if args.devices:
            world_size = len([d for d in args.devices.split(",") if d.strip()])
        else:
            world_size = _detect_gpu_count()
    else:
        world_size = args.world_size or 1
    world_size = max(1, world_size)

    scheduler_args = argparse.Namespace(
        runs_root=str(runs_root),
        paper_dir=str(paper_dir),
        lock_json=args.lock_json,
        budget_signal=args.budget_signal,
        groupby=args.groupby,
        experiment_name=experiment_name,
        dry_run=args.dry_run,
        budget_stop=args.budget_stop,
        max_concurrent=args.max_concurrent,
        world_size=world_size,
        devices=args.devices,
        resume=args.resume,
        seeds=seeds,
        dataset_mode=dataset_mode,
        real_data=real_data,
        hf_name=hf_name,
        seq_len=seq_len,
        batch_size=batch_size,
        steps=steps,
        cooldown=cooldown,
        epsilon_ucb=epsilon_ucb,
        config_path=config_path,
        retries=args.retries,
        offline_data_dir=offline_data_dir,
    )

    scheduler = Scheduler(scheduler_args)
    scheduler.prepare_jobs(baselines, methods)
    print(
        f"[scheduler] experiment={experiment_name} baselines={baselines} methods={methods} seeds={seeds} "
        f"max_concurrent={args.max_concurrent} world_size={world_size}"
    )
    scheduler.execute()
    print("[scheduler] completed.")


if __name__ == "__main__":
    main()
