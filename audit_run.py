"""Audit finished episodes using pinned author reductions and scalar affinities."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from check_scores import author_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / "plan.json").read_text())
    runtime = json.loads((run / "runtime.json").read_text())
    from runtime_config import checkpoint_catalog
    checkpoint = Path(next(e["local_dir"] for e in checkpoint_catalog()["checkpoints"] if e["repo_id"] == plan["checkpoint"]))
    statistics = json.loads((checkpoint / "dataset_statistics.json").read_text())[runtime["unnorm_key"]]["action"]
    config = json.loads((checkpoint / "config.json").read_text())
    bins = np.linspace(-1, 1, config["n_action_bins"])
    centers = (bins[:-1] + bins[1:]) / 2
    low, high = np.asarray(statistics["q01"]), np.asarray(statistics["q99"])
    mask = np.asarray(statistics["mask"], dtype=bool)
    begin, end = runtime["valid_action_token_range"]
    import hashlib
    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    deg = author_class("deg", ["compute_degmat_uncertainty"])
    eigv = author_class("eigv", ["compute_eigval_uncertainty"])
    ecc = author_class("ecc", ["compute_eccentricity_uncertainty"])
    ecc.thres = plan["ecc_eigenvalue_threshold"]
    kle = author_class("kle", ["laplacian_matrix", "heat_kernel", "normalize_kernel", "scale_entropy", "vn_entropy", "kle_score"])
    kle.t, kle.normalize, kle.scale, kle.jitter = plan["kle_heat_time"], True, True, 0.0
    def decode(ids):
        assert np.all((ids >= begin) & (ids < end))
        normalized = centers[np.clip(end - ids - 1, 0, len(centers) - 1)]
        raw = np.where(mask, 0.5 * (normalized + 1) * (high-low) + low, normalized)
        executed = raw.astype(np.float32)
        executed[..., 6] = -np.sign(2 * executed[..., 6] - 1)
        scoring = normalized.copy()
        scoring[..., 6] = executed[..., 6]
        return raw, executed, scoring
    def scalar_affinity(a, b):
        if a[6] != b[6]:
            return 0.0
        squared = sum((float(a[i]) - float(b[i])) ** 2 for i in range(6)) / 6
        return math.exp(-squared / (2 * plan["action_sigma"] ** 2))
    raw_records = (run / "episodes.jsonl").read_bytes() if (run / "episodes.jsonl").exists() else b""
    rows = [json.loads(line) for line in raw_records.splitlines(keepends=True) if line.endswith(b"\n")]
    errors = {method: 0.0 for method in plan["methods"]}
    queried = 0
    for row in rows:
        assert row["plan_sha256"] == digest(run / "plan.json")
        assert row["action_sha256"] == digest(run / row["action_path"])
        controls = np.load(run / row["action_path"], allow_pickle=False)
        assert controls.shape == (row["num_actions"], 7)
        expected_steps = list(range(0, len(controls), plan["score_stride"]))
        assert row["query_steps"] == expected_steps and row["score_queries"] == len(expected_steps)
        scores = []
        for step in expected_steps:
            artifact = (run / row["artifact_folder"]) / f"query_{step:04d}.npz"
            query = json.loads(artifact.with_suffix(".json").read_text())
            assert query["artifact_sha256"] == digest(artifact)
            with np.load(artifact, allow_pickle=False) as saved:
                base_raw, base_control, base_normalized = decode(saved["base_token_ids"])
                assert np.array_equal(base_raw, saved["base_action"])
                assert np.array_equal(base_control, controls[step])
                assert np.array_equal(base_normalized, saved["base_normalized"])
                for prefix in ("sampled", "perturbed"):
                    raw, _, normalized = decode(saved[prefix + "_token_ids"])
                    assert raw.shape == (5, 7)
                    assert np.array_equal(raw, saved[prefix + "_actions"])
                    assert np.array_equal(normalized, saved[prefix + "_normalized"])
                sampled, perturbed = saved["sampled_normalized"], saved["perturbed_normalized"]
                assert np.array_equal(saved["sampled_token_ids"][0], saved["perturbed_token_ids"][0])
                W = np.asarray([[scalar_affinity(a, b) for b in sampled] for a in sampled])
                assert np.allclose(W, saved["action_affinity"], rtol=1e-12, atol=1e-12)
                sim = [scalar_affinity(base_normalized, a) for a in perturbed]
                weights = saved["instruction_weights"]
                reference = {"deg": -deg.compute_degmat_uncertainty(W),
                             "eigv": -eigv.compute_eigval_uncertainty(W),
                             "ecc": -ecc.compute_eccentricity_uncertainty(W),
                             "kle": -kle.kle_score(W, 1-W, None),
                             "spuq": sum(float(w)*s for w,s in zip(weights, sim))/float(weights.sum())}
                for method in errors:
                    errors[method] = max(errors[method], abs(query["scores"][method] - reference[method]))
                    assert np.isclose(query["scores"][method], reference[method], rtol=1e-8, atol=1e-8)
                assert np.allclose(saved["confidence"], [reference[m] for m in plan["methods"]], atol=1e-8)
                scores.append(query["scores"])
                queried += 1
        for method in errors:
            assert np.isclose(row["scores"][method], np.mean([s[method] for s in scores]), rtol=1e-12, atol=1e-12)
            assert row["first_query_scores"][method] == scores[0][method]
    result = {"passed": True, "completed_episodes_audited": len(rows), "queries_audited": queried,
              "maximum_absolute_score_error": errors, "decoded_actions_match_executed_controls": True,
              "sampled_tokens_all_valid": True, "spuq_original_sample_shared": True,
              "limits": "Audits only completed episode artifacts; does not predict or reuse historical success labels."}
    (run / "artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
