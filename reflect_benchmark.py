"""Pinned LIBERO-Reflect tasks and static extraction of numeric initial states."""
import hashlib
import json
from pathlib import Path
import pickletools
import types
import zipfile

import numpy as np

from runtime_config import REFLECT_ROOT
DEFAULT_ROOT = REFLECT_ROOT
COMMIT = "3b8e6eb2501969b441fdaaf9da32c14ba77e4bfd"


def safe_initial_states(path):
    """Read the numeric ndarray payload; never evaluate pickle globals/reducers."""
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(members) != 1:
            raise ValueError(f"Unexpected initial-state archive: {path}")
        ops = list(pickletools.genops(archive.read(members[0])))
    shapes = []
    for i in range(len(ops) - 4):
        part = ops[i:i + 5]
        if (part[0][0].name == "MARK" and part[1][0].name.startswith("BININT")
                and part[1][1] == 1 and part[2][0].name.startswith("BININT")
                and part[3][0].name.startswith("BININT") and part[4][0].name == "TUPLE2"):
            shapes.append((part[2][1], part[3][1]))
    strings = [arg for op, arg, _ in ops
               if op.name in {"BINUNICODE", "SHORT_BINUNICODE"}]
    if len(shapes) != 1 or "f8" not in strings or "<" not in strings:
        raise ValueError(f"Unsupported numeric initial-state format: {path}")
    shape = shapes[0]
    if not 1 <= shape[0] <= 10000 or not 1 <= shape[1] <= 10000:
        raise ValueError(f"Invalid initial-state shape: {shape}")
    nbytes = shape[0] * shape[1] * 8
    payloads = []
    for op, arg, _ in ops:
        if op.name in {"BINBYTES", "SHORT_BINBYTES"} and len(arg) == nbytes:
            payloads.append(arg)
        elif op.name == "BINUNICODE" and len(arg) == nbytes and "latin1" in strings:
            payloads.append(arg.encode("latin1"))
    if len(payloads) != 1:
        raise ValueError(f"Ambiguous numeric initial-state payload: {path}")
    states = np.frombuffer(payloads[0], dtype="<f8").reshape(shape).copy()
    if not np.isfinite(states).all():
        raise ValueError(f"Non-finite initial states: {path}")
    return states


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_suite(benchmark, suite_name, mode="libero", root=DEFAULT_ROOT):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    if mode == "libero":
        return suite
    if mode != "libero_reflect":
        raise ValueError(f"Unknown benchmark mode: {mode}")
    root = Path(root).resolve()
    generated_path = root / "GENERATED_SCENES.json"
    generated = json.loads(generated_path.read_text()) if generated_path.exists() else None
    if generated is not None:
        if (generated.get("benchmark_commit") != COMMIT or not generated.get("complete")
                or generated.get("variant") != "author_generated_temp_swap"):
            raise ValueError("Incomplete or incompatible author-generated scenes")
        folder = suite_name + "_temp_swap"
        bddl_dir = root / "bddl_files" / folder
        init_dir = root / "init_files" / folder
        expected = generated["suites"][suite_name]["tasks"]
    else:
        expected = None
        folder = suite_name + "_swap"
        pkg = root / "reflect/libero/libero"
        bddl_dir = pkg / "bddl_files/bddl_files" / folder
        init_dir = pkg / "init_files/init_files" / folder
    if generated is None:
        provenance = json.loads((root / "DOWNLOAD_PROVENANCE.json").read_text())
        if provenance["commit"] != COMMIT:
            raise ValueError("LIBERO-Reflect commit mismatch")
    cache = {}
    new_tasks = []
    for i, task in enumerate(suite.tasks):
        bddl = bddl_dir / task.bddl_file
        initial = init_dir / task.init_states_file
        if not bddl.is_file() or not initial.is_file():
            raise ValueError(f"Missing pinned swap assets: {bddl}, {initial}")
        if expected is not None:
            checks = expected[task.name]
            if (file_sha256(bddl) != checks["bddl_sha256"]
                    or file_sha256(initial) != checks["init_file_sha256"]):
                raise ValueError(f"Author-generated asset hash mismatch: {task.name}")
        states = safe_initial_states(initial)
        if states.shape[0] != 50:
            raise ValueError(f"Expected 50 states for {task.name}")
        cache[i] = states
        new_tasks.append(task._replace(problem_folder=folder))
    suite.tasks = new_tasks
    suite.reflect_initial_states = cache
    suite.reflect_variant = "author_generated_temp_swap" if generated is not None else "swap"
    suite.reflect_manifest_sha256 = file_sha256(generated_path) if generated is not None else None
    suite.get_task_bddl_file_path = types.MethodType(
        lambda self, i: str(bddl_dir / self.tasks[i].bddl_file), suite)
    suite.reflect_asset_paths = {i: {"bddl": str(bddl_dir / task.bddl_file),
                                   "initial_states": str(init_dir / task.init_states_file)}
                                for i, task in enumerate(new_tasks)}
    return suite


def episode_provenance(suite, task_id, mode):
    value = {"benchmark_mode": mode,
             "nominal_split": "challenging" if mode == "libero_reflect" else "standard"}
    if mode == "libero_reflect":
        paths = suite.reflect_asset_paths[task_id]
        value.update(benchmark_commit=COMMIT, swap_asset_variant=suite.reflect_variant,
                     bddl_sha256=file_sha256(paths["bddl"]),
                     init_file_sha256=file_sha256(paths["initial_states"]))
        if suite.reflect_manifest_sha256 is not None:
            value["scene_manifest_sha256"] = suite.reflect_manifest_sha256
    return value
