"""Calibrate concurrent real rollouts, then finish a durable per-GPU queue."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

from formal_common import atomic_json


def finished_rows(run):
    path = Path(run) / "episodes.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_bytes().splitlines(keepends=True) if l.endswith(b"\n") and l.strip()]


def stage_rate(run, start, end):
    """Only use complete same-episode intervals wholly inside the timing window."""
    runtime = json.loads((run / "runtime.json").read_text())
    stride = json.loads((run / "plan.json").read_text())["score_stride"]
    origin = datetime.datetime.fromisoformat(runtime["started_at"]).timestamp()
    observations = []
    for line in (run / "worker.log").read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("phase") != "running" or "elapsed_sec" not in row:
            continue
        wall = origin + row["elapsed_sec"]
        if start <= wall <= end:
            observations.append(row)
    intervals = []
    for a, b in zip(observations, observations[1:]):
        key = lambda r: (r["split"], r["task_id"], r["episode_idx"])
        if key(a) == key(b) and b["actions"] - a["actions"] == stride:
            seconds = b["elapsed_sec"] - a["elapsed_sec"]
            if seconds > 0:
                intervals.append(seconds)
    return {"intervals": len(intervals), "seconds": sum(intervals),
            "actions_per_second": stride * len(intervals) / sum(intervals) if intervals else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("gpu", type=int)
    parser.add_argument("--stage-seconds", type=int, default=90)
    args = parser.parse_args()
    batch = args.batch.resolve()
    manifest = json.loads((batch / "manifest.json").read_text())
    lane = next(l for l in manifest["lanes"] if l["gpu"] == args.gpu)
    lock = (batch / f".gpu{args.gpu}.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Formal source changed: {name}")
    active = {}
    completed = []
    pending = list(lane["jobs"])
    for job in list(pending):
        final = batch / job["run"] / "finalization.json"
        if final.exists() and json.loads(final.read_text()).get("complete"):
            if len(finished_rows(batch / job["run"])) != job["expected_episodes"]:
                raise ValueError("Completed shard has an unexpected episode count")
            completed.append({"suite": job["suite"], "run": job["run"], "episodes": job["expected_episodes"]})
            pending.remove(job)
    capacity = min(lane["maximum_workers"], len(pending))
    status_path = batch / f"gpu{args.gpu}_queue.json"
    suspended = set()

    def status(phase, **extra):
        atomic_json(status_path, {"phase": phase, "gpu": args.gpu, "manager_pid": os.getpid(),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "active": [{"suite": j["suite"], "run": j["run"], "pid": p.pid,
                                "suspended_for_calibration": p.pid in suspended} for p, j in active.values()],
                    "pending": [{"suite": j["suite"], "run": j["run"]} for j in pending],
                    "completed": completed, **extra})

    def launch(job):
        run = (batch / job["run"])
        command = ["bash", str(ROOT / "run_formal_accelerated.sh"), "--run", str(run),
                   "--suite", job["suite"], "--stride", str(manifest["score_stride"]),
                   "--task-end", str(manifest["task_end"]),
                   "--initials"] + [str(i) for i in lane["initials"]]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=lane["gpu_uuid"], MUJOCO_EGL_DEVICE_ID=str(lane["egl"]))
        with (run / "worker.log").open("ab") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        previous = json.loads((run / "launch.json").read_text()) if (run / "launch.json").exists() else {}
        atomic_json(run / "launch.json", {"pid": process.pid, "run": str(run), "gpu": args.gpu,
                    "expected_episodes": job["expected_episodes"], "command": command, "attempt": previous.get("attempt", 0) + 1,
                    "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        active[process.pid] = (process, job)
        print("FORMAL_WORKER_STARTED", job["suite"], args.gpu, process.pid, flush=True)

    def resume_all():
        for pid in list(suspended):
            if pid in active and active[pid][0].poll() is None:
                os.kill(pid, signal.SIGCONT)
            suspended.discard(pid)

    try:
        if not pending:
            status("complete", calibration=False, concurrent_worker_limit=0)
            return
        # All four checkpoints run their real-weight and token/RNG checks before timing.
        for _ in range(capacity):
            launch(pending.pop(0))
        if capacity == 1 or any(job["expected_episodes"] < 20 for _, job in active.values()):
            limit = capacity
            status("running", calibration=False, concurrent_worker_limit=limit)
        else:
            status("warming", calibration=True)
            deadline = time.monotonic() + 1800
            while True:
                failed = [(p.pid, j["suite"], p.returncode) for p, j in active.values() if p.poll() is not None]
                if failed:
                    raise RuntimeError(f"Formal startup failed: {failed}")
                ready = all(((batch / j["run"]) / "artifacts/standard" / f"task1_init{lane['initials'][0]+1}" / "query_0000.png").exists()
                            for _, j in active.values())
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Four model workers did not finish initial validation")
                time.sleep(2)
            ordered = list(active.values())
            stages = []
            for count in sorted({1, min(2, capacity), capacity}):
                if count == 4:
                    # Give all resumed workers one score interval before starting the window.
                    settle = 25
                else:
                    settle = 15
                for index, (process, _) in enumerate(ordered):
                    if process.poll() is not None:
                        raise RuntimeError(f"Worker exited during calibration: {process.pid}")
                    if index < count and process.pid in suspended:
                        os.kill(process.pid, signal.SIGCONT)
                        suspended.remove(process.pid)
                    elif index >= count and process.pid not in suspended:
                        os.kill(process.pid, signal.SIGSTOP)
                        suspended.add(process.pid)
                status("calibrating", calibration=True, concurrent_workers=count, stage="settling")
                time.sleep(settle)
                started = time.time()
                status("calibrating", calibration=True, concurrent_workers=count, stage="measuring")
                for _ in range(args.stage_seconds):
                    if any(p.poll() is not None for p, _ in ordered):
                        raise RuntimeError("Worker exited during throughput measurement")
                    time.sleep(1)
                ended = time.time()
                measurements = [{"suite": job["suite"], **stage_rate((batch / job["run"]), started, ended)}
                                for _, job in ordered[:count]]
                valid = all(v["intervals"] >= 2 for v in measurements)
                rate = sum(v["actions_per_second"] for v in measurements) if valid else None
                stages.append({"concurrent_workers": count, "duration_sec": ended - started,
                               "interval_measurements": measurements, "valid": valid,
                               "combined_actions_per_second": rate})
                atomic_json(batch / f"gpu{args.gpu}_throughput.json", {"gpu": args.gpu, "stages": stages, "complete": False})
                print("THROUGHPUT_STAGE", json.dumps(stages[-1]), flush=True)
            valid = [s for s in stages if s["valid"]]
            if not valid:
                raise RuntimeError("No complete throughput measurement windows")
            best_rate = max(s["combined_actions_per_second"] for s in valid)
            # When measurements differ by less than 5%, keep higher concurrency.
            chosen = max((s for s in valid if s["combined_actions_per_second"] >= best_rate * .95),
                         key=lambda s: s["concurrent_workers"])
            limit = chosen["concurrent_workers"]
            calibration = {"gpu": args.gpu, "complete": True, "stages": stages,
                           "selected_workers": limit, "selected_actions_per_second": chosen["combined_actions_per_second"],
                           "limits": "Real rollout intervals; suites and episode lengths vary. Measures total rate, not identical-input latency. Warmup and suspended intervals are excluded."}
            atomic_json(batch / f"gpu{args.gpu}_throughput.json", calibration)
            resume_all()
            # Requeue excess workers without losing durable completed episodes.
            for process, job in ordered[limit:]:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                active.pop(process.pid)
                pending.append(job)
                atomic_json((batch / job["run"]) / "calibration_requeue.json",
                            {"preserved_episodes": len(finished_rows(batch / job["run"])), "reason": "selected faster concurrent-worker count"})
            status("running", calibration=False, concurrent_worker_limit=limit)
        while active or pending:
            while pending and len(active) < limit:
                launch(pending.pop(0))
            for pid, (process, job) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                run = (batch / job["run"])
                state = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
                if code != 0 or state.get("phase") != "complete":
                    raise RuntimeError(f"Worker failed for {job['suite']}, exit={code}, log={run/'worker.log'}")
                with (run / "finalizer.log").open("ab") as log:
                    subprocess.run([sys.executable, str(ROOT / "finish_report.py"), str(run)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                completed.append({"suite": job["suite"], "run": str(run), "episodes": len(finished_rows(run))})
                active.pop(pid)
            status("running", calibration=False, concurrent_worker_limit=limit)
            if active:
                time.sleep(10)
        status("complete", calibration=False, concurrent_worker_limit=limit)
    except BaseException as exc:
        resume_all()
        status("manager_failed", error=str(exc), worker_policy="unpaused live workers continue; no further queued jobs launched")
        raise


if __name__ == "__main__":
    main()
