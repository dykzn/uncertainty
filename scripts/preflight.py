"""CPU-only checks of pinned data, checkpoint completeness and score reductions."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_config import checkpoint_catalog, configure_runtime, REFLECT_ROOT
from scripts.download_assets import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", action="store_true")
    args = parser.parse_args()
    verify(REFLECT_ROOT)
    configure_runtime()
    from reflect_benchmark import safe_initial_states, make_suite
    from libero.libero import benchmark
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    for suite in suites:
        standard = make_suite(benchmark, suite)
        challenging = make_suite(benchmark, suite, "libero_reflect")
        assert len(standard.tasks) == len(challenging.tasks) == 10
        assert all(s.shape[0] == 50 for s in challenging.reflect_initial_states.values())
    if args.weights:
        for entry in checkpoint_catalog()["checkpoints"]:
            folder = Path(entry["local_dir"])
            meta = json.loads((folder / ".hfd/repo_metadata.json").read_text())
            assert meta["sha"] == entry["revision"]
            for file in meta["siblings"]:
                target = folder / file["rfilename"]
                size = file.get("size", (file.get("lfs") or {}).get("size"))
                assert target.is_file() and size is not None and target.stat().st_size == size, target
            assert not any(folder.rglob("*.part")) and not any(folder.rglob("*.aria2"))
            print("WEIGHT_FILES_VERIFIED", entry["directory"])
    from check_scores import main as check
    check()
    print("PREFLIGHT_PASSED", "four suites; 4000 distinct full-evaluation episodes")


if __name__ == "__main__":
    main()
