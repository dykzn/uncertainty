"""Report live per-suite metrics and audited completion without changing workers."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent

from formal_common import atomic_json, ranking_metrics
from action_scores import METHODS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    batch = args.batch.resolve()
    manifest = json.loads((batch / "manifest.json").read_text())
    while True:
        all_rows, seen, jobs = [], set(), []
        for lane in manifest["lanes"]:
            for job in lane["jobs"]:
                run = (batch / job["run"])
                path = run / "episodes.jsonl"
                rows = [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
                        if line.endswith(b"\n") and line.strip()] if path.exists() else []
                for row in rows:
                    key = (row["suite"], row["split"], row["task_id"], row["episode_idx"])
                    if key in seen:
                        raise ValueError(f"Duplicate full-evaluation episode: {key}")
                    if row["suite"] != job["suite"] or row["episode_idx"] - 1 not in lane["initials"]:
                        raise ValueError(f"Unexpected shard episode: {key}")
                    seen.add(key)
                all_rows.extend(rows)
                final_path = run / "finalization.json"
                audited = (json.loads(final_path.read_text()).get("artifact_audit_passed", False)
                           if final_path.exists() else False)
                jobs.append({"suite": job["suite"], "gpu": lane["gpu"], "run": str(run),
                             "completed_episodes": len(rows), "expected_episodes": job["expected_episodes"], "audited": audited})
        groups = {}
        metrics = []
        for suite in manifest["suites"]:
            suite_rows = [r for r in all_rows if r["suite"] == suite]
            groups[suite] = {}
            for split in ("standard", "challenging", "mixed"):
                chosen = suite_rows if split == "mixed" else [r for r in suite_rows if r["split"] == split]
                labels = [r["is_success"] for r in chosen]
                result = {"episodes": len(chosen), "successes": sum(labels),
                          "failures": len(labels) - sum(labels), "methods": {}}
                for method in METHODS:
                    m = ranking_metrics(labels, [r["scores"][method] for r in chosen]) if chosen else None
                    result["methods"][method] = m
                    numbers = [100 * m[k] for k in ("AUROC", "AUPR_average_precision", "FPR95_interpolated")] if m else [None]*3
                    metrics.append([suite, split, method, len(chosen), sum(labels), len(labels)-sum(labels), *numbers])
                groups[suite][split] = result
        complete = len(all_rows) == manifest["expected_episodes"] and all(j["audited"] for j in jobs)
        lanes = {str(l["gpu"]): (json.loads((batch / f"gpu{l['gpu']}_queue.json").read_text())
                                 if (batch / f"gpu{l['gpu']}_queue.json").exists() else {}) for l in manifest["lanes"]}
        failed = any(v.get("phase") == "manager_failed" for v in lanes.values())
        calibration = {}
        for lane in manifest["lanes"]:
            path = batch / f"gpu{lane['gpu']}_throughput.json"
            if path.exists():
                calibration[str(lane["gpu"])] = json.loads(path.read_text())
        atomic_json(batch / "summary.json", {"complete": complete, "completed_episodes": len(all_rows),
                    "expected_episodes": manifest["expected_episodes"], "positive_class": "success", "aggregation": "episode_mean",
                    "score_stride": manifest["score_stride"], "groups": groups})
        atomic_json(batch / "status.json", {"phase": "complete" if complete else "needs_attention" if failed else "running",
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "completed_episodes": len(all_rows), "expected_episodes": manifest["expected_episodes"], "shards": jobs,
                    "gpu_queues": lanes, "throughput_calibration": calibration})
        temporary = batch / ".metrics.csv.tmp"
        with temporary.open("w", newline="") as sink:
            writer = csv.writer(sink)
            writer.writerow(["suite", "cohort", "method", "episodes", "successes", "failures",
                             "AUROC_percent", "AUPR_AP_percent", "FPR95_percent"])
            writer.writerows(metrics)
        temporary.replace(batch / "metrics.csv")
        if complete:
            atomic_json(batch / "finalization.json", {"complete": True, "all_shards_audited": True,
                        "episodes": manifest["expected_episodes"], "metrics": str(batch / "metrics.csv")})
        if complete or not args.watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
