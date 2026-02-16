"""
Step 5: Create prototype-based strata for coverage-balanced video grouping.

Contains all functionality for:
- Building low-dimensional SVD embedding of video scores
- Selecting K prototypes via farthest-first traversal
- Assigning videos to nearest prototype buckets
- Producing strata for compilation selection
- Automatically choosing number of strata based on coverage metrics (n_strata="auto")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.core import set_seeds

# ============================================================
# Helper Function 1: Build embedding
# ============================================================


def build_embedding(
    video_scores: np.ndarray,
    k_embed: int = 16,
    normalize: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Build low-dimensional SVD embedding from video scores.

    Parameters
    ----------
    video_scores : np.ndarray
        Shape (n_videos, n_statements), already centered (from Step 4)
    k_embed : int
        Number of embedding dimensions
    normalize : bool
        If True, normalize rows to unit norm
    seed : int or None
        Random seed for SVD

    Returns
    -------
    Z : np.ndarray
        Shape (n_videos, k_embed), optionally normalized
    """
    n_videos, n_statements = video_scores.shape
    max_components = min(n_videos - 1, n_statements - 1, 128)
    max_components = max(1, max_components)

    k_embed = min(k_embed, max_components)
    k_embed = max(1, k_embed)

    svd = TruncatedSVD(n_components=k_embed, random_state=seed)
    Z = svd.fit_transform(video_scores)

    if normalize:
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    return Z.astype(np.float32)


# ============================================================
# Helper Function 2: Farthest-first prototypes
# ============================================================


def farthest_first_prototypes(
    Z: np.ndarray, K: int, init: str = "medoid", seed: int | None = None
) -> np.ndarray:
    """
    Select K prototypes from Z using farthest-first traversal.

    Parameters
    ----------
    Z : np.ndarray
        Shape (n_videos, k_embed), normalized (assumed)
    K : int
        Number of prototypes
    init : str
        Initialization: 'medoid' (default) or 'random'
    seed : int or None
        Random seed

    Returns
    -------
    prototype_indices : np.ndarray
        Shape (K,) indices into Z of selected prototypes
    """
    n = Z.shape[0]
    K = min(K, n)

    if seed is not None:
        np.random.seed(seed)

    # Choose initial prototype
    if init == "medoid":
        mean_vec = Z.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(Z - mean_vec, axis=1)
        first_idx = np.argmin(norms)
    elif init == "random":
        first_idx = np.random.randint(n)
    else:
        raise ValueError(f"Unknown init mode: {init}")

    prototypes = [first_idx]

    # Maintain minimum distance to any selected prototype
    d_min = np.full(n, np.inf)
    d_min[first_idx] = 0

    for _ in range(K - 1):
        # Compute distances from all points to the last-added prototype
        last_proto = prototypes[-1]
        dists = np.linalg.norm(Z - Z[last_proto], axis=1)

        # Update minimum distances
        d_min = np.minimum(d_min, dists)

        # Select point with largest minimum distance
        next_proto = np.argmax(d_min)
        prototypes.append(next_proto)

    return np.array(prototypes, dtype=int)


# ============================================================
# Helper Function 3: Assign to strata
# ============================================================


def assign_strata(
    Z: np.ndarray, prototype_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each video to nearest prototype stratum.

    Parameters
    ----------
    Z : np.ndarray
        Shape (n_videos, k_embed)
    prototype_indices : np.ndarray
        Shape (K,) indices of prototypes

    Returns
    -------
    stratum_labels : np.ndarray
        Shape (n_videos,) stratum assignment
    distances_to_prototype : np.ndarray
        Shape (n_videos,) distance to assigned prototype
    """
    n_videos = Z.shape[0]
    K = len(prototype_indices)

    prototype_points = Z[prototype_indices]

    dists = np.zeros((n_videos, K), dtype=np.float32)
    for k in range(K):
        dists[:, k] = np.linalg.norm(Z - prototype_points[k], axis=1)

    stratum_labels = np.argmin(dists, axis=1)
    distances_to_prototype = dists[np.arange(n_videos), stratum_labels]

    return stratum_labels.astype(int), distances_to_prototype.astype(np.float32)


# ============================================================
# Helper Function 4: Compute coverage metrics
# ============================================================


def compute_coverage_metrics(
    stratum_labels: np.ndarray, distances_to_proto: np.ndarray
) -> Dict:
    """
    Compute coverage tightness and balance metrics for a strata assignment.

    Parameters
    ----------
    stratum_labels : np.ndarray
        Shape (n_videos,) stratum assignment
    distances_to_proto : np.ndarray
        Shape (n_videos,) distance from each video to its assigned prototype

    Returns
    -------
    dict
        Keys: mean_dist, max_dist, p95_dist, min_size, singleton_frac, size_hist
    """
    mean_dist = float(np.mean(distances_to_proto))
    max_dist = float(np.max(distances_to_proto))
    p95_dist = float(np.percentile(distances_to_proto, 95))

    unique, counts = np.unique(stratum_labels, return_counts=True)
    sizes = counts.tolist()
    min_size = int(np.min(sizes)) if len(sizes) > 0 else 0
    singleton_frac = (
        float(np.sum(counts == 1) / len(counts)) if len(counts) > 0 else 0.0
    )

    size_hist = {int(u): int(c) for u, c in zip(unique, counts)}

    return {
        "mean_dist": mean_dist,
        "max_dist": max_dist,
        "p95_dist": p95_dist,
        "min_size": min_size,
        "singleton_frac": singleton_frac,
        "size_hist": size_hist,
    }


# ============================================================
# Helper Function 5: Automatic stratum count selection
# ============================================================


def choose_n_strata_auto(
    Z: np.ndarray,
    prototype_order: np.ndarray,
    min_stratum_size: int = 3,
    target_max_dist: float = 0.35,
    k_cap: int | None = None,
) -> Tuple[int, pd.DataFrame]:
    """
    Automatically select optimal number of strata via sweep and coverage-based rule.

    Parameters
    ----------
    Z : np.ndarray
        Shape (n_videos, k_embed), embedding
    prototype_order : np.ndarray
        Shape (K_cap,) prototype indices from farthest-first, in order
    min_stratum_size : int
        Minimum allowed stratum size (hard constraint)
    target_max_dist : float
        Target maximum distance to prototype (selection preference)
    k_cap : int or None
        Maximum number of strata to consider (length of prototype_order)

    Returns
    -------
    k_selected : int
        Selected number of strata
    sweep_df : pd.DataFrame
        Columns: K, mean_dist, max_dist, p95_dist, min_size, singleton_frac
    """
    n_videos = Z.shape[0]

    # Determine k_cap if not provided
    if k_cap is None:
        k_cap = min(n_videos, 20)
    k_cap = max(1, k_cap)
    k_cap = min(k_cap, len(prototype_order))

    # Sweep over K values
    sweep_records = []
    for K in range(1, k_cap + 1):
        proto_indices = prototype_order[:K]
        labels, dists = assign_strata(Z, proto_indices)
        metrics = compute_coverage_metrics(labels, dists)

        sweep_records.append(
            {
                "K": K,
                "mean_dist": metrics["mean_dist"],
                "max_dist": metrics["max_dist"],
                "p95_dist": metrics["p95_dist"],
                "min_size": metrics["min_size"],
                "singleton_frac": metrics["singleton_frac"],
            }
        )

    sweep_df = pd.DataFrame(sweep_records)

    # Selection rule: find feasible Ks where min_size >= min_stratum_size
    feasible = sweep_df[sweep_df["min_size"] >= min_stratum_size]

    if len(feasible) > 0:
        # Among feasible, prefer smallest K meeting target_max_dist
        meeting_target = feasible[feasible["max_dist"] <= target_max_dist]
        if len(meeting_target) > 0:
            k_selected = int(meeting_target.iloc[0]["K"])
        else:
            # No K meets target, pick feasible K with smallest max_dist
            k_selected = int(feasible.loc[feasible["max_dist"].idxmin(), "K"])
    else:
        # No feasible K: use fallback
        if n_videos < min_stratum_size:
            k_selected = 1
        else:
            k_selected = max(1, n_videos // min_stratum_size)

    return k_selected, sweep_df


def run(
    pipeline_context: Dict,
    n_strata: int | str = "auto",
    k_embed: int = 16,
    init: str = "medoid",
    normalize: bool = True,
    min_stratum_size: int = 3,
    target_max_dist: float = 0.35,
    max_singleton_frac: float = 0.0,
    k_cap: int | None = None,
    seed: int | None = None,
) -> Dict:
    """
    Step 5: Create prototype-based strata for video grouping.

    Parameters
    ----------
    pipeline_context : dict
        Pipeline state from previous step (contains video_scores, video_ids)
    n_strata : int | str
        Number of strata. If "auto", automatically determine based on coverage metrics.
        If int, use that value (clamped to valid range).
    k_embed : int
        Dimensionality of embedding
    init : str
        Prototype initialization: 'medoid' or 'random'
    normalize : bool
        Normalize embedding rows to unit norm
    min_stratum_size : int
        Minimum size of any stratum (hard constraint, default 3)
    target_max_dist : float
        Target maximum distance to prototype for auto selection (default 0.35)
    max_singleton_frac : float
        Deprecated/informational (with min_stratum_size=3, no singletons allowed)
    k_cap : int or None
        Max number of strata to consider for auto selection. If None, use min(n_videos, 20).
    seed : int or None
        Random seed

    Returns
    -------
    dict
        Updated pipeline context with strata and prototype information
    """
    if seed is not None:
        set_seeds(seed)

    output_dir = Path(pipeline_context["output_root"])
    video_scores = pipeline_context["video_scores"]

    # Get video_ids
    if "video_ids" in pipeline_context:
        video_ids = pipeline_context["video_ids"]
    elif "video_index_df" in pipeline_context:
        video_ids = pipeline_context["video_index_df"]["video_id"].tolist()
    else:
        raise ValueError("video_ids or video_index_df required in pipeline_context")

    n_videos, n_statements = video_scores.shape

    if n_videos < 1:
        raise ValueError("No videos in pipeline_context")

    # Validate video_ids length
    if len(video_ids) != n_videos:
        raise ValueError(f"len(video_ids)={len(video_ids)} != n_videos={n_videos}")

    # Determine k_cap for prototype sweep
    if k_cap is None:
        k_cap = min(n_videos, 20)
    k_cap = max(1, k_cap)

    # Build embedding
    Z = build_embedding(video_scores, k_embed=k_embed, normalize=normalize, seed=seed)

    # Compute prototype_order once, up to k_cap
    prototype_order = farthest_first_prototypes(Z, K=k_cap, init=init, seed=seed)

    # Determine final number of strata
    n_strata_mode = (
        "auto" if isinstance(n_strata, str) and n_strata == "auto" else "manual"
    )
    sweep_df = None

    if n_strata_mode == "auto":
        k_selected, sweep_df = choose_n_strata_auto(
            Z,
            prototype_order,
            min_stratum_size=min_stratum_size,
            target_max_dist=target_max_dist,
            k_cap=k_cap,
        )
        prototype_indices = prototype_order[:k_selected]
        selection_rule_used = (
            sweep_df[sweep_df["K"] == k_selected].iloc[0].to_dict()
            if len(sweep_df) > 0
            else {}
        )
    else:
        # Manual mode: use provided n_strata
        n_strata = int(n_strata)
        n_strata = min(n_strata, n_videos)
        n_strata = max(1, n_strata)
        prototype_indices = prototype_order[:n_strata]
        k_selected = n_strata
        selection_rule_used = None

    # Assign strata using final prototype_indices
    stratum_labels, distances_to_proto = assign_strata(Z, prototype_indices)

    # Compute coverage metrics
    coverage_metrics = compute_coverage_metrics(stratum_labels, distances_to_proto)

    # Get prototype video IDs
    prototype_video_ids = [video_ids[int(idx)] for idx in prototype_indices]

    # Compute stratum sizes
    unique, counts = np.unique(stratum_labels, return_counts=True)
    stratum_sizes = {int(u): int(c) for u, c in zip(unique, counts)}

    # Save artifacts
    np.save(output_dir / "embedding_Z.npy", Z)
    np.save(output_dir / "prototype_indices.npy", prototype_indices)

    prototype_df = pd.DataFrame(
        {
            "prototype_id": np.arange(len(prototype_indices)),
            "video_id": prototype_video_ids,
        }
    )
    prototype_df.to_csv(output_dir / "prototype_video_ids.csv", index=False)

    strata_df = pd.DataFrame(
        {
            "video_id": video_ids,
            "stratum_id": stratum_labels,
            "is_prototype": [
                1 if int(i) in prototype_indices else 0 for i in range(n_videos)
            ],
            "dist_to_prototype": distances_to_proto,
        }
    )
    strata_df.to_csv(output_dir / "strata.csv", index=False)

    # Save sweep results if auto mode
    if n_strata_mode == "auto" and sweep_df is not None:
        sweep_df.to_csv(output_dir / "strata_k_sweep.csv", index=False)

    # Build report
    report = {
        "n_videos": int(n_videos),
        "n_statements": int(n_statements),
        "k_embed": int(k_embed),
        "n_strata": int(k_selected),
        "n_strata_mode": n_strata_mode,
        "init_mode": init,
        "normalize": bool(normalize),
        "min_stratum_size": int(min_stratum_size),
        "seed": seed,
        "stratum_sizes": stratum_sizes,
        "coverage_metrics": {
            "mean_dist": float(coverage_metrics["mean_dist"]),
            "max_dist": float(coverage_metrics["max_dist"]),
            "p95_dist": float(coverage_metrics["p95_dist"]),
            "min_size": int(coverage_metrics["min_size"]),
            "singleton_frac": float(coverage_metrics["singleton_frac"]),
        },
    }

    # Add auto-specific fields if applicable
    if n_strata_mode == "auto":
        report["target_max_dist"] = float(target_max_dist)
        report["k_cap"] = int(k_cap)
        if selection_rule_used and isinstance(selection_rule_used, dict):
            report["selected_k_metrics"] = {
                "K": int(selection_rule_used.get("K", k_selected)),
                "mean_dist": float(selection_rule_used.get("mean_dist", 0.0)),
                "max_dist": float(selection_rule_used.get("max_dist", 0.0)),
                "p95_dist": float(selection_rule_used.get("p95_dist", 0.0)),
                "min_size": int(selection_rule_used.get("min_size", 0)),
            }

    with open(output_dir / "strata_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return {
        **pipeline_context,
        "embedding_Z": Z,
        "prototype_indices": prototype_indices.tolist(),
        "prototype_video_ids": prototype_video_ids,
        "stratum_labels": stratum_labels,
        "n_strata": int(k_selected),
        "k_embed": k_embed,
    }


__all__ = [
    "run",
    "build_embedding",
    "farthest_first_prototypes",
    "assign_strata",
    "compute_coverage_metrics",
    "choose_n_strata_auto",
]
