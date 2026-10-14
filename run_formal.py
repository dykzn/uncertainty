"""Full-suite greedy OpenVLA rollouts with shared action uncertainty samples."""
import argparse
import datetime
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
from runtime_config import PROJECT_ROOT, OPENVLA_SOURCE, configure_runtime, checkpoint_catalog
from formal_common import atomic_json, append_record, initial_hash, query_seed, ranking_metrics
from reflect_benchmark import DEFAULT_ROOT, make_suite, safe_initial_states, file_sha256, episode_provenance
from action_scores import METHODS, query_scores


def records(path):
    """Ignore an incomplete final append; never accept duplicate episodes."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        end = raw.rfind(b"\n") + 1
        with path.open("r+b") as sink:
            sink.truncate(end)
        raw = raw[:end]
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    keys = [(r["split"], r["task_id"], r["episode_idx"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate completed episodes")
    return rows


def summarize(run, rows, expected):
    groups = {}
    for split in ("standard", "challenging", "mixed"):
        selected = rows if split == "mixed" else [r for r in rows if r["split"] == split]
        labels = [r["is_success"] for r in selected]
        groups[split] = {"episodes": len(selected), "successes": sum(labels),
                         "failures": len(labels) - sum(labels), "methods": {}}
        if not selected:
            continue
        for method in METHODS:
            groups[split]["methods"][method] = {
                "episode_mean": ranking_metrics(labels, [r["scores"][method] for r in selected]),
                "first_query": ranking_metrics(labels, [r["first_query_scores"][method] for r in selected]),
            }
    atomic_json(run / "summary.json", {
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "complete": len(rows) == expected, "expected_episodes": expected,
        "completed_episodes": len(rows), "positive_class": "success", "groups": groups,
        "limits": "Episode means are retrospective trajectory scores. "
                  "First-query scores use a fixed observation time. Report single-split and mixed cohorts separately.",
    })


class VisionCache:
    """Reuse the same visual features only within one unchanged observation."""
    def __init__(self, vision):
        self.original = vision.forward
        self.active = False
        self.clear()
        vision.forward = self.forward

    def clear(self):
        self.pixels = self.features = None

    def forward(self, pixels):
        import torch
        if not self.active:
            return self.original(pixels)
        if self.features is None:
            self.pixels = pixels.detach().clone()
            self.features = self.original(pixels)
        elif not torch.equal(pixels, self.pixels):
            raise ValueError("Attempted to reuse visual features for a different image")
        return self.features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--suite", choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"), required=True)
    parser.add_argument("--splits", choices=("standard", "challenging"), nargs="+", default=["standard", "challenging"])
    parser.add_argument("--initials", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--task-end", type=int, default=10)
    parser.add_argument("--max-actions", type=int, default=520)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if (args.stride < 1 or not 1 <= args.task_end <= 10 or args.samples != 5
            or len(args.splits) != len(set(args.splits))):
        raise ValueError("Invalid evaluation configuration")
    if len(args.initials) != len(set(args.initials)) or not all(0 <= i < 50 for i in args.initials):
        raise ValueError("Invalid initial-state indices")
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / ".runner.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    checks = json.loads((ROOT / "score_checks.json").read_text())
    if not checks.get("passed"):
        raise ValueError("Passing author reduction checks are required")
    instructions_path = ROOT / "assets" / f"{args.suite.removeprefix('libero_')}_instructions.json"
    instruction_data = json.loads(instructions_path.read_text())
    instructions = {r["task_id"] - 1: r for r in instruction_data["tasks"]}
    catalog = checkpoint_catalog()["checkpoints"]
    checkpoint_entry = next(e for e in catalog if e["model"] == "OpenVLA"
                            and args.suite.removeprefix("libero_") in e["libero_suites"])
    checkpoint = Path(checkpoint_entry["local_dir"])
    if instruction_data["suite"] != args.suite:
        raise ValueError("Instruction suite mismatch")
    jobs = [(task, initial, split) for task in range(args.task_end)
            for initial in args.initials for split in args.splits]
    plan = {
        "protocol": "openvla_action_space_blackbox_portable_v2", "model": "openvla",
        "suite": args.suite, "checkpoint": checkpoint_entry["repo_id"], "checkpoint_revision": checkpoint_entry["revision"],
        "methods": list(METHODS), "splits": args.splits,
        "jobs": [{"task_id": t + 1, "episode_idx": i + 1, "split": s} for t, i, s in jobs],
        "environment_seed": 7, "initial_wait_actions": 10,
        "maximum_policy_actions": args.max_actions, "score_stride": args.stride,
        "graph_samples": args.samples, "sampling_temperature": 1.0,
        "trajectory_sampling": "Every action is executed; uncertainty is computed at step % score_stride == 0",
        "top_k": 0, "top_p": 1.0, "sampling_vocabulary": "valid action-token band only",
        "executed_policy": "original unrestricted greedy OpenVLA; auxiliary samples never executed",
        "action_affinity": "exp(-mean(squared normalized six-coordinate differences)/(2*sigma^2)) * gripper agreement",
        "action_sigma": args.sigma, "ecc_eigenvalue_threshold": 0.9, "kle_heat_time": 0.3,
        "spuq_perturbations": "original instruction + four reviewed paraphrases; original uses shared sample[0]",
        "spuq_instruction_weights": "MiniLM normalized-embedding cosine similarity",
        "spuq_output_similarity": "action RBF replaces text embedding similarity",
        "graph_relation": "action affinity replaces text NLI; KLE entail=W, contradiction=1-W, neutral=0",
        "interpretation": "Action-space adaptations, not unchanged QA implementations",
        "instructions_sha256": file_sha256(instructions_path),
        "author_sources": json.loads((ROOT / "author_sources/manifest.json").read_text()),
        "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in
                          (ROOT / "action_scores.py", Path(__file__).resolve(),
                           ROOT / "openvla_inference.py", OPENVLA_SOURCE / "experiments/robot/openvla_utils.py",
                           OPENVLA_SOURCE / "prismatic/extern/hf/modeling_prismatic.py")},
    }
    plan_path = run / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Resume configuration or source mismatch")
    atomic_json(plan_path, plan)
    episode_path = run / "episodes.jsonl"
    rows = records(episode_path)
    completed = {(r["task_id"] - 1, r["episode_idx"] - 1, r["split"]) for r in rows}
    expected_jobs = set(jobs)
    if not completed <= expected_jobs:
        raise ValueError("Unexpected resumed episodes")
    for row in rows:
        if row["plan_sha256"] != file_sha256(plan_path) or file_sha256(run / row["action_path"]) != row["action_sha256"]:
            raise ValueError("Completed episode provenance mismatch")
    summarize(run, rows, len(jobs))
    configure_runtime()
    import tensorflow as tf
    import torch
    import yaml
    from PIL import Image
    from transformers import LogitsProcessor
    from openvla_inference import load_backend, load_policy, prepare_libero_rendering
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    torch.set_num_threads(4)
    backend, _ = load_backend("openvla")
    prepare_libero_rendering()
    image_utils = importlib.import_module("experiments.robot.libero.libero_utils")
    robot_utils = importlib.import_module("experiments.robot.robot_utils")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    suites = {split: make_suite(benchmark, args.suite,
                              "libero" if split == "standard" else "libero_reflect", DEFAULT_ROOT)
              for split in args.splits}
    paths = yaml.safe_load((Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml").read_text())
    cfg = SimpleNamespace(pretrained_checkpoint=str(checkpoint), task_suite_name=args.suite, unnorm_key="")
    model, processor, _, _ = load_policy(backend, "openvla", cfg)
    if model.get_action_dim(cfg.unnorm_key) != 7:
        raise ValueError("Expected seven action dimensions")
    vocab_end = model.vocab_size
    vocab_begin = vocab_end - model.config.n_action_bins

    class ActionOnly(LogitsProcessor):
        def __call__(self, input_ids, scores):
            masked = torch.full_like(scores, float("-inf"))
            masked[:, vocab_begin:vocab_end] = scores[:, vocab_begin:vocab_end]
            return masked

    cache = VisionCache(model.vision_backbone)
    sampling = {"do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
                "logits_processor": [ActionOnly()]}

    def prepare_image(pixels):
        converted = tf.image.convert_image_dtype(tf.convert_to_tensor(pixels), tf.float32)
        cropped = backend.crop_and_resize(converted, 0.9, 1)
        cropped = tf.image.convert_image_dtype(tf.clip_by_value(cropped, 0, 1), tf.uint8, saturate=True)
        return Image.fromarray(cropped.numpy()).convert("RGB")

    def inputs_for(image, instruction):
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        return processor(prompt, image).to(backend.DEVICE, dtype=torch.bfloat16)

    def predict(inputs, seed=None):
        kwargs = {"do_sample": False} if seed is None else sampling
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        raw, ids = model.predict_action(**inputs, unnorm_key=cfg.unnorm_key,
                                        return_action_token_ids=True, **kwargs)
        ids = np.asarray(ids, dtype=np.int64)
        if ids.shape != (7,) or not np.all((ids >= vocab_begin) & (ids < vocab_end)):
            raise ValueError(f"Invalid action tokens: {ids}")
        raw = np.asarray(raw, dtype=np.float64)
        executed = robot_utils.invert_gripper_action(
            robot_utils.normalize_gripper_action(raw.astype(np.float32).copy(), binarize=True))
        indices = np.clip(vocab_end - ids - 1, 0, len(model.bin_centers) - 1)
        normalized = np.r_[model.bin_centers[indices[:6]], executed[6]]
        return raw, executed, normalized, ids

    runtime = {"pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"), "torch": torch.__version__,
               "attention_backend": "eager", "checkpoint": str(checkpoint),
               "weight_loading": model.formal_loading_info, "unnorm_key": cfg.unnorm_key,
               "valid_action_token_range": [vocab_begin, vocab_end],
               "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(run / "runtime.json", runtime)
    print("BLACKBOX_POLICY_READY", json.dumps({"suite": args.suite, "stride": args.stride}), flush=True)
    validated = False
    started = time.monotonic()
    for task_id, initial, split in jobs:
        if (task_id, initial, split) in completed:
            continue
        suite = suites[split]
        task = suite.get_task(task_id)
        instruction = instructions[task_id]
        if instruction["original"] != task.language:
            raise ValueError("Instruction and benchmark task disagree")
        states = (suite.reflect_initial_states[task_id] if split == "challenging" else
                  safe_initial_states(Path(paths["init_states"]) / task.problem_folder / task.init_states_file))
        bddl = (suite.get_task_bddl_file_path(task_id) if split == "challenging" else
                str(Path(paths["bddl_files"]) / task.problem_folder / task.bddl_file))
        folder = run / "artifacts" / split / f"task{task_id + 1}_init{initial + 1}"
        folder.mkdir(parents=True, exist_ok=True)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        status = {"phase": "running", "pid": os.getpid(), "split": split,
                  "task_id": task_id + 1, "episode_idx": initial + 1,
                  "completed_episodes": len(rows), "expected_episodes": len(jobs), "actions": 0}
        atomic_json(run / "status.json", status)
        actions, score_rows, raw_score_rows, query_steps = [], [], [], []
        episode_started = time.monotonic()
        success = False
        try:
            env.seed(7)
            np.random.seed(query_seed(args.suite, task_id, initial, -1))
            env.reset()
            obs = env.set_init_state(states[initial])
            for _ in range(10):
                obs, _, _, _ = env.step([0.] * 6 + [-1.])
            for step in range(args.max_actions):
                seed = query_seed(args.suite, task_id, initial, step)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                pixels = image_utils.get_libero_image(obs, 224)
                image = prepare_image(pixels)
                base_inputs = inputs_for(image, task.language)
                cache.clear()
                cache.active = True
                with torch.inference_mode():
                    base_raw, executed, base_normalized, base_ids = predict(base_inputs)
                    if not validated:
                        cache.active = False
                        native = backend.get_vla_action(model, processor, str(checkpoint),
                            {"full_image": pixels}, task.language, cfg.unnorm_key, center_crop=True)
                        uncached = predict(base_inputs)
                        if not np.array_equal(base_raw, native) or not np.array_equal(base_ids, uncached[3]):
                            raise ValueError("Preprocessing or visual caching changed greedy action")
                        sample_reference = predict(base_inputs, seed + 101)
                        cache.active = True
                        sample_cached = predict(base_inputs, seed + 101)
                        if not np.array_equal(sample_reference[3], sample_cached[3]):
                            raise ValueError("Visual caching changed sampled action")
                        atomic_json(run / "real_weight_checks.json", {
                            "passed": True, "native_greedy_action_bit_equal": True,
                            "cached_greedy_tokens_bit_equal": True, "cached_sampled_tokens_bit_equal": True,
                            "all_checked_tokens_in_action_band": True, "image_crop_area": 0.9,
                            "base_action": base_raw.tolist(), "base_token_ids": base_ids.tolist()})
                        print("REAL_WEIGHT_CHECKS_PASSED", flush=True)
                        validated = True
                        if args.validate_only:
                            atomic_json(run / "status.json", {**status, "phase": "validated"})
                            return
                    if step % args.stride == 0:
                        sample_values = [predict(base_inputs, query_seed(args.suite, task_id, initial, step,
                                         seed=700 + k)) for k in range(args.samples)]
                        # The original-instruction SPUQ sample is shared with the graph methods.
                        perturbed_values = [sample_values[0]] + [
                            predict(inputs_for(image, text), query_seed(args.suite, task_id, initial, step,
                                    seed=800 + k)) for k, text in enumerate(instruction["perturbations"][1:])]
                        normalized_samples = np.stack([v[2] for v in sample_values])
                        normalized_perturbed = np.stack([v[2] for v in perturbed_values])
                        confidence, raw_scores, W, similarities = query_scores(normalized_samples, base_normalized,
                            normalized_perturbed, instruction["instruction_weights"], sigma=args.sigma)
                        artifact = folder / f"query_{step:04d}.npz"
                        with artifact.with_suffix(".tmp").open("wb") as sink:
                            np.savez_compressed(sink, step=step, base_action=base_raw, base_normalized=base_normalized,
                                base_token_ids=base_ids, sampled_actions=np.stack([v[0] for v in sample_values]),
                                sampled_token_ids=np.stack([v[3] for v in sample_values]),
                                sampled_normalized=normalized_samples, perturbed_actions=np.stack([v[0] for v in perturbed_values]),
                                perturbed_token_ids=np.stack([v[3] for v in perturbed_values]),
                                perturbed_normalized=normalized_perturbed, action_affinity=W,
                                instruction_weights=np.asarray(instruction["instruction_weights"]), spuq_similarities=similarities,
                                method_names=np.asarray(METHODS), confidence=np.asarray([confidence[m] for m in METHODS]))
                        artifact.with_suffix(".tmp").replace(artifact)
                        image.save(folder / f"query_{step:04d}.png")
                        atomic_json(folder / f"query_{step:04d}.json", {
                            "step": step, "scores": confidence, "raw_scores": raw_scores,
                            "artifact_sha256": file_sha256(artifact), "sampling_seeds": [
                                query_seed(args.suite, task_id, initial, step, seed=700 + k)
                                for k in range(args.samples)],
                            "paraphrase_seeds": [query_seed(args.suite, task_id, initial, step, seed=800 + k)
                                                 for k in range(4)]})
                        score_rows.append(confidence)
                        raw_score_rows.append(raw_scores)
                        query_steps.append(step)
                        status.update(actions=step, score_queries=len(score_rows),
                                      elapsed_sec=time.monotonic() - started, latest_scores=confidence)
                        atomic_json(run / "status.json", status)
                        print(json.dumps(status), flush=True)
                cache.clear()
                actions.append(executed.copy())
                obs, _, done, _ = env.step(executed.tolist())
                if done:
                    success = True
                    break
            action_path = folder / "executed_actions.npy"
            np.save(action_path, np.stack(actions))
            row = {"suite": args.suite,
                   **episode_provenance(suite, task_id, "libero" if split == "standard" else "libero_reflect"),
                   "task_id": task_id + 1, "episode_idx": initial + 1, "split": split,
                   "is_success": bool(success), "num_actions": len(actions), "score_queries": len(score_rows),
                   "query_steps": query_steps, "scores": {m: float(np.mean([r[m] for r in score_rows])) for m in METHODS},
                   "first_query_scores": score_rows[0], "raw_first_query_scores": raw_score_rows[0],
                   "initial_state_sha256": initial_hash(states[initial]), "bddl_sha256": file_sha256(bddl),
                   "plan_sha256": file_sha256(plan_path), "action_path": str(action_path.relative_to(run)),
                   "action_sha256": file_sha256(action_path), "artifact_folder": str(folder.relative_to(run)),
                   "elapsed_sec": time.monotonic() - episode_started}
            append_record(episode_path, row)
            rows.append(row)
            summarize(run, rows, len(jobs))
            print("EPISODE_COMPLETE", json.dumps(row), flush=True)
        finally:
            cache.clear()
            cache.active = False
            env.close()
    atomic_json(run / "status.json", {"phase": "complete", "completed_episodes": len(rows),
                                       "expected_episodes": len(jobs), "elapsed_sec": time.monotonic() - started})
    print("BLACKBOX_FORMAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
