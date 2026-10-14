"""Fetch a pinned benchmark snapshot; verify the exact 80 task/state pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_config import REFLECT_ROOT, configure_runtime

REVISION = "3b8e6eb2501969b441fdaaf9da32c14ba77e4bfd"


def verify(root):
    inventory = json.loads((ROOT / "assets/formal_inventory.json").read_text())
    for suite, meta in inventory.items():
        for task in meta["tasks"]:
            name = task["task_name"]
            for split, base, folder in [("standard", root / "standard/libero/libero", suite),
                                        ("reflect", root / "reflect/libero/libero", suite + "_swap")]:
                bddl = base / ("bddl_files" if split == "standard" else "bddl_files/bddl_files") / folder / (name + ".bddl")
                initial = base / ("init_files" if split == "standard" else "init_files/init_files") / folder / (name + ".pruned_init")
                for p, key in [(bddl, split + "_bddl_sha256"), (initial, split + "_initial_states_sha256")]:
                    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != task[key]:
                        raise ValueError(f"Benchmark hash mismatch: {suite}/{name}/{p.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not REFLECT_ROOT.exists() and not args.verify_only:
        import requests
        cache = ROOT / "runtime/downloads"
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / (REVISION + ".tar.gz")
        session = requests.Session()
        session.trust_env = False
        if not archive.exists():
            part = archive.with_suffix(".part")
            with session.get(f"https://codeload.github.com/aniri15/LIBERO-REFLECT/tar.gz/{REVISION}", stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with part.open("wb") as sink:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            sink.write(chunk)
            part.replace(archive)
        staging = REFLECT_ROOT.with_name(REFLECT_ROOT.name + ".extracting")
        staging.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as source:
            for member in source:
                parts = Path(member.name).parts[1:]
                if not parts:
                    continue
                if member.issym() or member.islnk() or ".." in parts or Path(*parts).is_absolute():
                    raise ValueError("Unsupported archive member")
                target = staging.joinpath(*parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        verify(staging)
        (staging / "DOWNLOAD_PROVENANCE.json").write_text(json.dumps({"repository": "https://github.com/aniri15/LIBERO-REFLECT", "commit": REVISION,
            "download_method": "pinned commit archive without proxies", "verification": "80 BDDL files and 80 numeric initial-state files match stored SHA256"}, indent=2) + "\n")
        staging.replace(REFLECT_ROOT)
    verify(REFLECT_ROOT)
    configure_runtime()
    print("BENCHMARK_READY", REVISION, "four suites, 50 initial states per task")


if __name__ == "__main__":
    main()
