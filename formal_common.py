"""Durable records and success-positive ranking metrics."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

def query_seed(suite, task, initial, query, seed=7):
    """Stable seed independent of worker ordering and interrupted/resumed episodes."""
    key = f"{seed}:{suite}:{task}:{initial}:{query}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") & 0x7fffffff

def initial_hash(state):
    return hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)

def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as sink:
        sink.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())

def ranking_metrics(labels, scores):
    """Success-positive, ties grouped; return null when one class is absent."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    p, n = int(y.sum()), int((~y).sum())
    if not p or not n:
        return None
    if not np.isfinite(s).all():
        raise ValueError("Non-finite scores")
    order = np.argsort(-s, kind="stable")
    sorted_s, sorted_y = s[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_s)), len(s) - 1]
    tp = np.cumsum(sorted_y)[ends]
    fp = ends + 1 - tp
    tpr = np.r_[0., tp / p]
    fpr = np.r_[0., fp / n]
    precision = tp / (tp + fp)
    precision_endpoint = np.r_[1., precision]
    return {"AUROC": float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)),
            "AUPR_average_precision": float(np.sum(np.diff(tpr) * precision)),
            "AUPR_trapezoidal": float(np.sum(np.diff(tpr) * (precision_endpoint[:-1] + precision_endpoint[1:]) / 2)),
            "FPR95_interpolated": float(np.interp(.95, tpr, fpr))}
