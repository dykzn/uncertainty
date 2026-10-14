"""Full-suite evaluation with validated greedy and multinomial decode graphs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

import run_formal as collector
import openvla_inference
from reflect_benchmark import file_sha256
from sampled_decode_graphs import SampledDecodeGraphs


def main():
    import torch
    score_stride = int(sys.argv[sys.argv.index("--stride")+1]) if "--stride" in sys.argv else 20
    audit_all_actions = "--audit-all-actions" in sys.argv
    if audit_all_actions:
        sys.argv.remove("--audit-all-actions")
    run = Path(sys.argv[sys.argv.index("--run")+1]).resolve()
    profile_path = run / "sampling_graph_speed_profile.json"
    if not profile_path.exists():
        import shutil
        run.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "validation/reference_decode_graphs.json", profile_path)
    profile = json.loads(profile_path.read_text())
    if not profile["passed"] or profile["sampled_sequences"] < 72:
        raise ValueError("Passing sampled token, RNG and score checks are required")
    for name in ("sampled_decode_graphs.py", "graphed_decode.py"):
        if file_sha256(ROOT/name) != profile["source_sha256"][name]:
            raise ValueError(f"Validated source changed: {name}")
    metadata = {"generation_accelerator": "native_eager_cuda_graph_greedy_and_sampling_v2",
                "sampling_decode_source_sha256": file_sha256(ROOT/"sampled_decode_graphs.py"),
                "graphed_decode_source_sha256": file_sha256(ROOT/"graphed_decode.py"),
                "sampling_graph_validation_sha256": file_sha256(run/"sampling_graph_speed_profile.json"),
                "sampling_generation": "native full-vocabulary softmax and multinomial captured with preserved RNG",
                "scored_query_greedy_validation": ("native token comparison at every environment action" if audit_all_actions
                                                    else "native token comparison at every scored query"),
                "sample_validation": "first sample at new prompt lengths and every 100 environment actions"}
    original_load = openvla_inference.load_policy
    original_seed = collector.query_seed
    original_append, original_json = collector.append_record, collector.atomic_json
    engines = []

    def seed(suite, task, initial, query, seed=7):
        if seed == 7 and query >= 0:
            for engine in engines:
                engine.audit_greedy = audit_all_actions or query % score_stride == 0
                if query == 0 and engine.current_task != (suite, task):
                    # Bound graph memory across long jobs with many prompt lengths.
                    import torch
                    torch.cuda.synchronize()
                    engine.graphs.clear()
                    engine.sample_graphs.clear()
                    torch.cuda.empty_cache()
                    engine.current_task = (suite, task)
                engine.sample_audit_key = (suite, task, initial, query) if query % 100 == 0 else None
        return original_seed(suite,task,initial,query,seed=seed)

    def load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        model = result[0]
        engine = SampledDecodeGraphs(model)
        engine.current_task = None
        engine.audit_greedy = True
        engine.sample_audit_key = None
        engine.greedy_lengths, engine.sample_lengths = set(), set()
        engine.disabled_greedy, engine.disabled_sample = set(), set()
        engine.audited_sample_keys = set()
        engine.greedy_checks = engine.sample_checks = engine.fallbacks = 0
        engines.append(engine)

        def generate(input_ids, **options):
            config = options.get("generation_config",model.generation_config)
            sampled = options.get("do_sample",config.do_sample)
            length = input_ids.shape[1]
            if length in (engine.disabled_sample if sampled else engine.disabled_greedy):
                return engine.original_generate(input_ids,**options)
            audit = (length not in engine.sample_lengths or
                     engine.sample_audit_key is not None and engine.sample_audit_key not in engine.audited_sample_keys
                     if sampled else engine.audit_greedy or length not in engine.greedy_lengths)
            if sampled and audit:
                cpu_before, cuda_before = torch.get_rng_state(), torch.cuda.get_rng_state()
            fast = engine.generate(input_ids,**options)
            if audit:
                if sampled:
                    cpu_after, cuda_after = torch.get_rng_state(), torch.cuda.get_rng_state()
                    torch.set_rng_state(cpu_before);torch.cuda.set_rng_state(cuda_before)
                native = engine.original_generate(input_ids,**options)
                equal = torch.equal(fast,native)
                if sampled:
                    equal = equal and torch.equal(cpu_after,torch.get_rng_state()) and torch.equal(cuda_after,torch.cuda.get_rng_state())
                if not equal:
                    (engine.disabled_sample if sampled else engine.disabled_greedy).add(length)
                    engine.fallbacks += 1
                    print("DECODE_GRAPH_FALLBACK",{"sampled":sampled,"length":length},flush=True)
                    return native
                if sampled:
                    engine.sample_lengths.add(length)
                    engine.sample_checks += 1
                    if engine.sample_audit_key is not None:
                        engine.audited_sample_keys.add(engine.sample_audit_key)
                else:
                    engine.greedy_lengths.add(length)
                    engine.greedy_checks += 1
            return fast

        model.generate = generate
        return result

    def annotate_json(path,value):
        if Path(path).name in ("runtime.json","status.json"):
            value={**value,**metadata}
        return original_json(path,value)

    def annotate_append(path,value):
        value={**value,**metadata}
        if engines:
            e=engines[0]
            value.update(graph_native_greedy_comparisons=e.greedy_checks,
                         graph_native_sample_and_rng_comparisons=e.sample_checks,
                         graph_disabled_greedy_lengths=sorted(e.disabled_greedy),
                         graph_disabled_sample_lengths=sorted(e.disabled_sample))
            original_json(run/"sampling_graph_runtime_checks.json",{**metadata,
                "native_greedy_comparisons_passed":e.greedy_checks,
                "native_sample_and_rng_comparisons_passed":e.sample_checks,
                "validated_greedy_lengths":sorted(e.greedy_lengths),"validated_sample_lengths":sorted(e.sample_lengths),
                "disabled_greedy_lengths":sorted(e.disabled_greedy),"disabled_sample_lengths":sorted(e.disabled_sample),
                "fallbacks":e.fallbacks})
        return original_append(path,value)

    openvla_inference.load_policy=load
    collector.query_seed=seed
    collector.atomic_json,collector.append_record=annotate_json,annotate_append
    collector.main()


if __name__=="__main__":main()
