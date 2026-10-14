"""Pinned local OpenVLA inference without training-only imports."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from runtime_config import PROJECT_ROOT, OPENVLA_SOURCE, checkpoint_catalog
ROOT = PROJECT_ROOT

def prepare_libero_rendering():
    """Honor independent EGL numbering without changing the Torch policy GPU.

    robosuite 1.4.1 checks CUDA visibility against an EGL ordinal once at import.
    Initialize CUDA with the real policy GPU, expose the EGL ordinal for that
    import-only check, then restore CUDA visibility before creating environments.
    Its EGL context subsequently uses MUJOCO_EGL_DEVICE_ID directly.
    """
    import torch
    torch.cuda.init()
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    egl = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if egl is None:
        raise ValueError("An explicit host EGL ordinal is required")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = egl
        importlib.import_module("robosuite.utils.binding_utils")
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original

def load_backend(variant):
    # The shared environment's flash_attn wheel has an incompatible Torch ABI.
    # These evaluations explicitly use eager attention. Prevent Transformers
    # 4.40 from eagerly importing that optional wheel while defining Llama.
    import transformers.utils
    import transformers.utils.import_utils
    transformers.utils.is_flash_attn_2_available = lambda: False
    transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
    if variant != "openvla":
        raise ValueError("This project evaluates OpenVLA")
    repo = OPENVLA_SOURCE
    sys.path.insert(0, str(repo))
    for name in ("prismatic", "prismatic.models", "prismatic.training", "prismatic.vla"):
        package = types.ModuleType(name)
        package.__path__ = [str(repo / name.replace(".", "/"))]
        package.__package__ = name
        sys.modules[name] = package
    source = repo / "experiments/robot/openvla_utils.py"
    # Import experiments as ordinary packages before installing the helper alias.
    importlib.import_module("experiments.robot")
    spec = importlib.util.spec_from_file_location("experiments.robot.openvla_utils", source)
    backend = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = backend
    spec.loader.exec_module(backend)
    return backend, repo

def load_policy(backend, variant, cfg):
    import torch
    checkpoint = Path(cfg.pretrained_checkpoint)
    metadata = json.loads((checkpoint / ".hfd/repo_metadata.json").read_text())
    catalog = checkpoint_catalog()["checkpoints"]
    entry = next(c for c in catalog if Path(c["local_dir"]) == checkpoint)
    if metadata["sha"] != entry["revision"] or any(checkpoint.rglob("*.aria2")):
        raise ValueError("Checkpoint revision mismatch or incomplete download")
    for sibling in metadata["siblings"]:
        file = checkpoint / sibling["rfilename"]
        expected_size = sibling.get("size", sibling.get("lfs", {}).get("size"))
        if not file.is_file() or expected_size is None or file.stat().st_size != expected_size:
            raise ValueError(f"Incomplete checkpoint file: {file}")
    # Explicit local classes keep attention support without rewriting checkpoints
    # or depending on the checkpoint's remote auto_map modeling files.
    config = backend.OpenVLAConfig.from_pretrained(cfg.pretrained_checkpoint, local_files_only=True)
    config.auto_map = None
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    model, loading_info = backend.OpenVLAForActionPrediction.from_pretrained(
        cfg.pretrained_checkpoint, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="eager", low_cpu_mem_usage=True, local_files_only=True,
        output_loading_info=True,
    )
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Pretrained state-dict mismatch: {loading_info}")
    model = model.to("cuda").eval()
    model.formal_loading_info = loading_info
    model.formal_attention_semantics = "causal"
    if type(model.language_model.model.layers[0].self_attn).__name__ != "LlamaAttention":
        raise ValueError("Formal scoring requires the explicitly selected eager Llama attention backend")
    model.norm_stats = json.loads((Path(cfg.pretrained_checkpoint) / "dataset_statistics.json").read_text())
    key = cfg.task_suite_name
    if key not in model.norm_stats:
        key += "_no_noops"
    if key not in model.norm_stats:
        raise KeyError(f"No normalization statistics for {cfg.task_suite_name}: {list(model.norm_stats)}")
    cfg.unnorm_key = key
    processor = load_processor(backend, cfg.pretrained_checkpoint)
    action_head = proprio_projector = None
    return model, processor, action_head, proprio_projector

def load_processor(backend, checkpoint):
    from transformers import LlamaTokenizerFast
    image_processor = backend.PrismaticImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = LlamaTokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    return TokenAlignedProcessor(backend.PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer))

class TokenAlignedProcessor:
    """Append OpenVLA's required empty token to both IDs and the mask.

    predict_action otherwise appends 29871 to input_ids alone. Eager Llama then
    sees a one-token-short attention mask; SDPA's mask elision can hide this bug.
    Preparing the same token here leaves the prompt content unchanged.
    """
    def __init__(self, processor):
        self.processor = processor

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        import torch
        inputs = self.processor(*args, **kwargs)
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        if ids.shape != mask.shape:
            raise ValueError("Processor returned mismatched token IDs and attention mask")
        if not torch.all(ids[:, -1] == 29871):
            inputs["input_ids"] = torch.cat((ids, ids.new_full((ids.shape[0], 1), 29871)), dim=1)
            inputs["attention_mask"] = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), dim=1)
        return inputs
