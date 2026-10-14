"""Wait for the detached worker, audit its artifacts, and write metric tables."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    launch = json.loads((run / "launch.json").read_text())
    while True:
        status_path = run / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("phase") == "complete":
            break
        try:
            os.kill(launch["pid"], 0)
        except ProcessLookupError:
            failure = {"complete": False, "worker_exited_before_completion": True,
                       "last_status": status, "worker_log": str(run / "worker.log")}
            (run / "finalization.json").write_text(json.dumps(failure, indent=2) + "\n")
            print(json.dumps(failure), flush=True)
            return
        time.sleep(15)
    subprocess.run([sys.executable, str(ROOT / "audit_run.py"), str(run)], check=True)
    summary = json.loads((run / "summary.json").read_text())
    with (run / "metrics.csv").open("w", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["cohort", "aggregation", "method", "episodes", "successes", "failures",
                         "AUROC_percent", "AUPR_AP_percent", "FPR95_percent"])
        for cohort, group in summary["groups"].items():
            for method, aggregations in group["methods"].items():
                for aggregation, metrics in aggregations.items():
                    numbers = ([100 * metrics[k] for k in
                                ("AUROC", "AUPR_average_precision", "FPR95_interpolated")]
                               if metrics is not None else [None] * 3)
                    writer.writerow([cohort, aggregation, method, group["episodes"], group["successes"],
                                     group["failures"], *numbers])
    final = {"complete": True, "artifact_audit_passed": True,
             "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "episodes": summary["completed_episodes"], "metrics": str(run / "metrics.csv")}
    (run / "finalization.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
