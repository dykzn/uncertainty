"""Download pinned public Hugging Face checkpoints, with resume and no proxies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_config import CHECKPOINT_ROOT, checkpoint_catalog


def download(session, url, path, expected_size=None, sha256=None):
    if path.is_file() and (expected_size is None or path.stat().st_size == expected_size):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        with part.open("ab" if offset else "wb") as sink:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    sink.write(chunk)
    if expected_size is not None and part.stat().st_size != expected_size:
        raise ValueError(f"Incomplete checkpoint file: {path.name}")
    if sha256:
        digest = hashlib.sha256()
        with part.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError(f"Checkpoint content hash mismatch: {path.name}")
    part.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["spatial", "object", "goal", "10", "all"], default="all")
    parser.add_argument("--embedding", action="store_true", help="only fetch the optional instruction encoder")
    args = parser.parse_args()
    import requests
    session = requests.Session()
    session.trust_env = False
    if args.embedding:
        repo = "sentence-transformers/all-MiniLM-L6-v2"
        revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "vocab.txt", "model.safetensors", "1_Pooling/config.json"]
        response = session.get(f"https://huggingface.co/api/models/{repo}/revision/{revision}", params={"blobs": "true"}, timeout=60)
        response.raise_for_status()
        metadata = response.json()
        assert metadata["sha"] == revision
        siblings = {e["rfilename"]: e for e in metadata["siblings"]}
        directory = ROOT / "assets/all-MiniLM-L6-v2"
        records = []
        for name in files:
            info = siblings[name]
            lfs = info.get("lfs") or {}
            path = directory / name
            download(session, f"https://huggingface.co/{repo}/resolve/{revision}/{name}", path,
                     info.get("size", lfs.get("size")), lfs.get("sha256"))
            records.append({"path": name, "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (directory / "download_manifest.json").write_text(json.dumps({"model": repo, "revision": revision,
            "proxy_used": False, "files": records}, indent=2) + "\n")
        print("INSTRUCTION_ENCODER_READY", revision)
        return
    for entry in checkpoint_catalog()["checkpoints"]:
        if args.suite != "all" and args.suite not in entry["libero_suites"]:
            continue
        response = session.get(f"https://huggingface.co/api/models/{entry['repo_id']}/revision/{entry['revision']}", params={"blobs": "true"}, timeout=60)
        response.raise_for_status()
        metadata = response.json()
        if metadata["sha"] != entry["revision"]:
            raise ValueError("Checkpoint revision mismatch")
        directory = Path(entry["local_dir"])
        for sibling in metadata["siblings"]:
            name = sibling["rfilename"]
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid checkpoint filename")
            lfs = sibling.get("lfs") or {}
            size = sibling.get("size", lfs.get("size"))
            if size is None:
                raise ValueError(f"Missing size metadata for {name}")
            print("CHECKPOINT_FILE", entry["directory"], name, flush=True)
            download(session, f"https://huggingface.co/{entry['repo_id']}/resolve/{entry['revision']}/{name}", directory / relative, size, lfs.get("sha256"))
        marker = directory / ".hfd/repo_metadata.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(metadata, indent=2) + "\n")
        print("CHECKPOINT_READY", entry["directory"], flush=True)


if __name__ == "__main__":
    main()
