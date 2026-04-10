"""Standalone helper for inspecting debug dumps in a notebook or script.

Usage:

    from app.infrastructure.debug.debug_tools import load_iteration, list_iterations

    # List all iterations for a session
    iters = list_iterations("data/debug/abc123")
    for meta in iters:
        print(meta["iteration"], meta["n_clusters"], meta["n_outliers"])

    # Load one iteration
    it = load_iteration("data/debug/abc123/iter_0007")
    print(it["meta"])
    print(it["M"].shape)
    import matplotlib.pyplot as plt
    plt.scatter(it["projection"][:, 0], it["projection"][:, 1], c=it["clusters"])
    plt.show()

This module deliberately has no dependencies on Flask, services, or session
state -- so you can import it in any Python environment that has numpy.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np


PathLike = Union[str, Path]


def list_iterations(session_dump_dir: PathLike) -> List[Dict[str, Any]]:
    """Return meta.json contents for every iteration dumped under a session dir."""
    session_dir = Path(session_dump_dir)
    if not session_dir.exists():
        return []

    out = []
    for iter_dir in sorted(session_dir.iterdir()):
        if not iter_dir.is_dir():
            continue
        meta_file = iter_dir / "meta.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def load_iteration(iter_dir: PathLike) -> Dict[str, Any]:
    """Load everything from a single iteration directory into a dict."""
    iter_dir = Path(iter_dir)
    if not iter_dir.exists():
        raise FileNotFoundError(f"No such directory: {iter_dir}")

    result: Dict[str, Any] = {}

    meta_file = iter_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            result["meta"] = json.load(f)

    for fname, key in [
        ("M.npy", "M"),
        ("clusters.npy", "clusters"),
        ("outliers.npy", "outliers"),
        ("projection.npy", "projection"),
    ]:
        path = iter_dir / fname
        if path.exists():
            result[key] = np.load(path)

    scores_path = iter_dir / "scores.npz"
    if scores_path.exists():
        with np.load(scores_path) as npz:
            result["scores"] = {k: npz[k] for k in npz.files}

    labels_path = iter_dir / "labels.json"
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            result["labels"] = json.load(f)

    return result


def diff_iterations(iter_dir_a: PathLike, iter_dir_b: PathLike) -> Dict[str, Any]:
    """Compute simple differences between two iterations.

    Useful for answering 'what changed when I applied constraint X'.
    """
    a = load_iteration(iter_dir_a)
    b = load_iteration(iter_dir_b)

    diff: Dict[str, Any] = {}

    if "M" in a and "M" in b:
        diff["M_frobenius_diff"] = float(np.linalg.norm(a["M"] - b["M"]))

    if "clusters" in a and "clusters" in b:
        if a["clusters"].shape == b["clusters"].shape:
            diff["n_relabeled"] = int(np.sum(a["clusters"] != b["clusters"]))

    if "outliers" in a and "outliers" in b:
        if a["outliers"].shape == b["outliers"].shape:
            diff["n_outlier_flips"] = int(np.sum(a["outliers"] != b["outliers"]))

    if "scores" in a and "scores" in b:
        score_diffs = {}
        for key in a["scores"]:
            if key in b["scores"] and a["scores"][key].shape == b["scores"][key].shape:
                delta = b["scores"][key] - a["scores"][key]
                score_diffs[key] = {
                    "mean_abs_delta": float(np.mean(np.abs(delta))),
                    "max_abs_delta": float(np.max(np.abs(delta))),
                }
        diff["score_deltas"] = score_diffs

    return diff
