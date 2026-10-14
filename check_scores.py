"""Check reductions independently against the pinned author functions."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.linalg import eigh
from action_scores import action_affinity, graph_uncertainties, query_scores

ROOT = Path(__file__).resolve().parent


def author_class(name, selected):
    source = ROOT / "author_sources/src/method" / (name + ".py")
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    namespace = {"np": np, "scipy": scipy, "eigh": eigh}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    result = namespace[cls.name]()
    return result


def main():
    deg = author_class("deg", ["compute_degmat_uncertainty"])
    eigv = author_class("eigv", ["compute_eigval_uncertainty"])
    ecc = author_class("ecc", ["compute_eccentricity_uncertainty"])
    ecc.thres = 0.9
    kle = author_class("kle", ["laplacian_matrix", "heat_kernel", "normalize_kernel", "scale_entropy", "vn_entropy", "kle_score"])
    kle.t, kle.normalize, kle.scale, kle.jitter = 0.3, True, True, 0.0
    rng = np.random.default_rng(241014)
    matrices = [np.ones((5, 5)), np.eye(5)]
    for _ in range(100):
        actions = np.c_[rng.normal(0, rng.uniform(0.01, 0.4), (5, 6)), rng.choice([-1, 1], 5)]
        matrices.append(action_affinity(actions))
    errors = {m: 0.0 for m in ("deg", "eigv", "ecc", "kle")}
    for W in matrices:
        ours = graph_uncertainties(W)
        reference = {"deg": deg.compute_degmat_uncertainty(W),
                     "eigv": eigv.compute_eigval_uncertainty(W),
                     "ecc": ecc.compute_eccentricity_uncertainty(W),
                     "kle": kle.kle_score(W, 1-W, None)}
        for method in errors:
            errors[method] = max(errors[method], abs(ours[method] - reference[method]))
            assert np.isclose(ours[method], reference[method], rtol=1e-9, atol=1e-9), (method, ours, reference)
    coherent, diverse = graph_uncertainties(np.ones((5, 5))), graph_uncertainties(np.eye(5))
    assert all(diverse[m] > coherent[m] for m in coherent)
    base = np.r_[np.zeros(6), 1.0]
    samples = np.tile(base, (5, 1))
    perturbed = samples.copy()
    perturbed[1, 0] = 0.2
    perturbed[3, 6] = -1
    weights = np.array([1, 0.8, 0.9, 0.7, 0.95])
    conf, raw, _, similarities = query_scores(samples, base, perturbed, weights)
    reference_spuq = sum(float(w) * float(s) for w, s in zip(weights, similarities)) / sum(weights)
    assert conf["spuq"] == raw["spuq"] and abs(raw["spuq"] - reference_spuq) < 1e-12
    assert similarities[3] == 0 and all(conf[m] == -raw[m] for m in ("deg", "eigv", "ecc", "kle"))
    result = {"passed": True, "author_revision": json.loads((ROOT / "author_sources/manifest.json").read_text())["revision"],
              "reference_matrices": len(matrices), "maximum_absolute_error": errors,
              "spuq_weighted_aggregation_error": abs(raw["spuq"] - reference_spuq),
              "coherent_and_disconnected_graph_ordering": True, "gripper_disagreement_similarity_zero": True}
    (ROOT / "score_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
