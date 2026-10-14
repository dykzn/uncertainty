"""Action-space adaptations of the published black-box UE reductions.

The relation function is new: an RBF on six normalized action coordinates,
multiplied by agreement of the executed, binarized gripper command. The graph
reductions retain the author implementation; SPUQ retains its weighted mean.
No success labels enter these functions.
"""
import numpy as np
from scipy.linalg import eigh, expm

METHODS = ("deg", "eigv", "ecc", "kle", "spuq")


def action_affinity(left, right=None, sigma=0.1):
    left = np.asarray(left, dtype=np.float64)
    right = left if right is None else np.asarray(right, dtype=np.float64)
    for values in (left, right):
        if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
            raise ValueError("Expected finite N x 7 normalized actions")
        if not np.isin(values[:, 6], (-1.0, 1.0)).all():
            raise ValueError("Gripper must be the executed discrete -1/+1 command")
    if sigma <= 0:
        raise ValueError("Positive action bandwidth required")
    squared = np.mean((left[:, None, :6] - right[None, :, :6]) ** 2, axis=-1)
    return np.exp(-squared / (2 * sigma ** 2)) * (left[:, None, 6] == right[None, :, 6])


def graph_uncertainties(W, ecc_threshold=0.9, kle_time=0.3):
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or len(W) < 2:
        raise ValueError("Expected a square action affinity matrix with at least two samples")
    if not np.isfinite(W).all() or np.min(W) < 0 or np.max(W) > 1 + 1e-12:
        raise ValueError("Invalid affinity entries")
    if not np.allclose(W, W.T, atol=1e-12) or not np.allclose(np.diag(W), 1):
        raise ValueError("Symmetric affinity and unit diagonal required")
    n = len(W)
    degrees = W.sum(axis=1)
    inv_sqrt = 1 / np.sqrt(degrees)
    normalized_L = np.eye(n) - inv_sqrt[:, None] * W * inv_sqrt[None, :]
    eigenvalues, eigenvectors = eigh(normalized_L)
    selected = eigenvectors[:, eigenvalues < ecc_threshold].T
    centered_norms = np.linalg.norm(selected - selected.mean(axis=1, keepdims=True), axis=1)
    # The author's KLE uses entail + 0.5 * neutral. Our continuous action
    # relation sets entail=W, contradiction=1-W, neutral=0, so its graph is W.
    L = np.diag(W.sum(axis=0)) - W
    K = expm(-kle_time * L)
    diagonal = np.sqrt(np.diag(K)) + 1e-12
    K = K / np.outer(diagonal, diagonal) / n
    kernel_eigenvalues = np.linalg.eigvalsh(K)
    if kernel_eigenvalues.min() < -1e-10:
        raise ValueError("Non-positive-semidefinite heat kernel")
    retained = kernel_eigenvalues[kernel_eigenvalues > 1e-8]
    return {
        "deg": float(1 - W.sum() / n ** 2),
        "eigv": float(np.maximum(0, 1 - eigenvalues).sum()),
        "ecc": float(np.linalg.norm(centered_norms)),
        "kle": float(-(retained * np.log(retained)).sum() / np.log(n)),
    }


def query_scores(sample_actions, base_action, perturbed_actions, instruction_weights, sigma=0.1):
    W = action_affinity(sample_actions, sigma=sigma)
    raw = graph_uncertainties(W)
    weights = np.asarray(instruction_weights, dtype=np.float64)
    similarities = action_affinity(np.asarray(base_action)[None, :], perturbed_actions, sigma=sigma)[0]
    if weights.shape != similarities.shape or not np.isfinite(weights).all() or np.min(weights) < 0:
        raise ValueError("Invalid SPUQ instruction weights")
    if weights.sum() <= 0:
        raise ValueError("SPUQ weights must have positive mass")
    raw["spuq"] = float(np.dot(weights, similarities) / weights.sum())
    # Consistent success-positive orientation fixed before collecting labels.
    confidence = {name: (-raw[name] if name != "spuq" else raw[name]) for name in METHODS}
    return confidence, raw, W, similarities
