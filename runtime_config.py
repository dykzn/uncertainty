"""Portable paths for code, benchmark assets and separately downloaded weights."""
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
OPENVLA_SOURCE = PROJECT_ROOT / "vendor/openvla"
LIBERO_SOURCE = PROJECT_ROOT / "vendor/libero"
REFLECT_ROOT = Path(os.environ.get("LIBERO_REFLECT_ROOT", PROJECT_ROOT / "external/LIBERO-REFLECT")).expanduser().resolve()
LIBERO_DATA_ROOT = Path(os.environ.get("LIBERO_DATA_ROOT", REFLECT_ROOT / "standard/libero/libero")).expanduser().resolve()
CHECKPOINT_ROOT = Path(os.environ.get("OPENVLA_CHECKPOINT_ROOT", PROJECT_ROOT / "weights/openvla")).expanduser().resolve()


def checkpoint_catalog():
    value = json.loads((PROJECT_ROOT / "config/checkpoints.json").read_text())
    for entry in value["checkpoints"]:
        entry["local_dir"] = str(CHECKPOINT_ROOT / entry["directory"])
    return value


def configure_runtime():
    """Create project-local LIBERO configuration before importing the simulator."""
    import yaml
    for name in ("bddl_files", "init_files", "assets"):
        if not (LIBERO_DATA_ROOT / name).is_dir():
            raise FileNotFoundError(f"Missing benchmark assets: {LIBERO_DATA_ROOT / name}; run scripts/download_assets.py or set LIBERO_DATA_ROOT")
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", PROJECT_ROOT / "runtime/libero_config")).expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    datasets = PROJECT_ROOT / "runtime/datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    wanted = {"benchmark_root": str(LIBERO_DATA_ROOT), "bddl_files": str(LIBERO_DATA_ROOT / "bddl_files"),
              "init_states": str(LIBERO_DATA_ROOT / "init_files"), "assets": str(LIBERO_DATA_ROOT / "assets"),
              "datasets": str(datasets)}
    config = config_dir / "config.yaml"
    if config.exists() and yaml.safe_load(config.read_text()) != wanted:
        # An explicit external config may belong to another project.
        if "LIBERO_CONFIG_PATH" in os.environ and config_dir != (PROJECT_ROOT / "runtime/libero_config").resolve():
            raise ValueError("LIBERO_CONFIG_PATH contains a different configuration; choose a separate config directory")
    temporary = config.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(wanted))
    temporary.replace(config)
    # Some original object loaders resolve meshes relative to their Python package.
    link = LIBERO_SOURCE / "libero/libero/assets"
    target = LIBERO_DATA_ROOT / "assets"
    if link.is_symlink() and link.resolve() != target.resolve():
        link.unlink()
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
    if link.resolve() != target.resolve():
        raise ValueError("Vendored simulator asset directory conflicts with LIBERO_DATA_ROOT")
    sys.path.insert(0, str(LIBERO_SOURCE))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return wanted
