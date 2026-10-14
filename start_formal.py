"""Start or resume a background batch using a configurable set of NVIDIA GPUs."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
from formal_common import atomic_json
from runtime_config import configure_runtime
from scripts.detect_gpus import detect


def partition(count, lanes):
    q, r = divmod(count, lanes)
    start, result = 0, []
    for index in range(lanes):
        end = start + q + (index < r)
        result.append(list(range(start, end)))
        start = end
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0", help="comma-separated nvidia-smi indices")
    parser.add_argument("--max-workers", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--initial-count", type=int, default=50)
    parser.add_argument("--task-end", type=int, default=10)
    parser.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    args = parser.parse_args()
    if args.stride < 1 or not 1 <= args.initial_count <= 50 or not 1 <= args.task_end <= 10:
        parser.error("invalid trajectory or scoring settings")
    wanted = [int(i) for i in args.gpus.split(",")]
    if len(wanted) != len(set(wanted)) or len(wanted) > args.initial_count:
        parser.error("GPU indices must be distinct and no more numerous than initial states")
    available = {c["index"]: c for c in detect()}
    cards = [available[i] for i in wanted]
    if any(c["egl"] is None for c in cards):
        parser.error("selected GPU does not provide an EGL renderer")
    suites = args.suites.split(",")
    allowed = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    if not set(suites) <= allowed or len(set(suites)) != len(suites):
        parser.error("invalid suites")
    sources = ["run_formal.py", "run_formal_accelerated.py", "run_formal_accelerated.sh", "formal_common.py",
               "runtime_config.py", "reflect_benchmark.py", "openvla_inference.py", "action_scores.py",
               "graphed_decode.py", "sampled_decode_graphs.py", "manage_formal_gpu.py", "aggregate_formal.py",
               "audit_run.py", "finish_report.py", "check_scores.py"]
    if args.resume:
        batch = args.resume.resolve()
        manifest = json.loads((batch / "manifest.json").read_text())
        for lane in manifest["lanes"]:
            for job in lane["jobs"]:
                with (batch / job["run"] / ".runner.lock").open("a") as lock:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise RuntimeError("A shard is still running; stop its old worker before resuming")
        if len(manifest["lanes"]) != len(cards):
            parser.error("resume requires the same shard count; GPU indices may change")
        for name, expected in manifest["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Resume source mismatch: {name}")
        for lane, card in zip(manifest["lanes"], cards):
            lane.update(gpu=card["index"], gpu_uuid=card["uuid"], egl=card["egl"], maximum_workers=args.max_workers)
    else:
        batch = (args.output or ROOT / "outputs/formal" / f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%d_%H%M%S}").resolve()
        if batch.exists() and any(batch.iterdir()):
            raise ValueError("Output directory is not empty; use --resume for an existing batch")
        batch.mkdir(parents=True, exist_ok=True)
        lanes = []
        for index, (card, initials) in enumerate(zip(cards, partition(args.initial_count, len(cards)))):
            jobs = []
            for suite in suites:
                relative = f"{suite}/shard{index}"
                run = batch / relative
                run.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / "validation/reference_decode_graphs.json", run / "sampling_graph_speed_profile.json")
                jobs.append({"suite": suite, "run": relative,
                             "expected_episodes": len(initials) * args.task_end * 2})
            lanes.append({"gpu": card["index"], "gpu_uuid": card["uuid"], "egl": card["egl"], "initials": initials,
                          "maximum_workers": args.max_workers, "jobs": jobs})
        manifest = {"protocol": "openvla_action_space_blackbox_portable_v2", "suites": suites,
                    "methods": ["deg", "eigv", "ecc", "kle", "spuq"], "score_stride": args.stride,
                    "task_end": args.task_end, "maximum_policy_actions": 520,
                    "expected_episodes": len(suites) * args.initial_count * args.task_end * 2,
                    "lanes": lanes, "source_sha256": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sources}}
        snapshots = batch / "source_snapshots"
        snapshots.mkdir()
        for name in sources:
            shutil.copy2(ROOT / name, snapshots / name)
    configure_runtime()
    atomic_json(batch / "manifest.json", manifest)
    launch = {"batch": str(batch), "expected_episodes": manifest["expected_episodes"],
              "score_stride": manifest["score_stride"], "processes": []}
    for lane in manifest["lanes"]:
        with (batch / f"gpu{lane['gpu']}_manager.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, str(ROOT / "manage_formal_gpu.py"), str(batch), str(lane["gpu"])],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch["processes"].append({"role": "gpu_manager", "gpu": lane["gpu"], "pid": process.pid})
    with (batch / "aggregator.log").open("ab") as log:
        process = subprocess.Popen([sys.executable, str(ROOT / "aggregate_formal.py"), str(batch), "--watch"],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch["processes"].append({"role": "aggregator", "pid": process.pid})
    atomic_json(batch / "launch.json", launch)
    atomic_json(ROOT / "outputs/latest_formal.json", launch)
    print(json.dumps(launch, indent=2))


if __name__ == "__main__":
    main()
