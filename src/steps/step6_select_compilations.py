"""
Step 6: Select compilations from clusters.

Contains all functionality for:
- Using k-center algorithm to select diverse anchor videos
- Greedily building compilations around each anchor
- Respecting duration constraints per compilation
- Balancing theme coherence, diversity, and cluster representation
- Saving compilation metadata and selection metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from src.core import set_seeds, ensure_dir, get_compilations_dir

# ============================================================
# K-Center & Selection Functions
# ============================================================


def kcenter_greedy(
    X: np.ndarray, k: int, spread_method: str = "euclidean", seed: int | None = None
) -> np.ndarray:
    """Select k diverse points using greedy k-center algorithm."""
    if seed is not None:
        np.random.seed(seed)

    n = X.shape[0]
    if k >= n:
        return np.arange(n)

    centers = [np.random.randint(n)]

    for _ in range(k - 1):
        dists = np.full(n, np.inf)
        for c in centers:
            if spread_method == "euclidean":
                d = np.linalg.norm(X - X[c], axis=1)
            elif spread_method == "cosine":
                d = cosine_distances(X[c : c + 1], X).flatten()
            else:
                raise ValueError(f"Unsupported metric: {spread_method}")
            dists = np.minimum(dists, d)
        next_center = np.argmax(dists)
        centers.append(next_center)

    return np.array(centers)


def select_compilations(
    embedding_Z: np.ndarray,
    stratum_labels: np.ndarray,
    video_ids: List[str],
    video_durations: Dict[str, float],
    n_compilations: int = 10,
    comp_min_seconds: float = 55.0,
    comp_max_seconds: float = 65.0,
    anchor_spread_method: str = "euclidean",
    theme_coherence_power: float = 1.0,
    diversity_power: float = 1.0,
    seed: int = 1234,
) -> List[Dict]:
    """Select balanced compilation segments via k-center anchors and diversity scoring."""
    np.random.seed(seed)

    n_strata = len(np.unique(stratum_labels))
    Z_norm = (embedding_Z - embedding_Z.mean(axis=0)) / (embedding_Z.std(axis=0) + 1e-8)

    anchor_idx = kcenter_greedy(
        Z_norm, n_compilations, spread_method=anchor_spread_method, seed=seed
    )

    D = cosine_distances(Z_norm, Z_norm)
    usage_count = np.zeros(len(Z_norm), dtype=int)

    compilations: List[Dict] = []

    for ci, a_idx in enumerate(anchor_idx):
        rng = np.random.default_rng(seed + ci)
        selected: List[Dict] = []
        stratum_counts = {s: 0 for s in range(n_strata)}
        total_seconds = 0.0
        available = set(range(len(Z_norm)))

        strata_order = list(range(n_strata))
        rng.shuffle(strata_order)

        for stratum in strata_order:
            stratum_indices = [i for i in available if stratum_labels[i] == stratum]
            if not stratum_indices:
                continue

            candidates = []
            for i in stratum_indices:
                vid_id = video_ids[i]
                dur = video_durations.get(vid_id, 0.0)
                if (
                    dur <= 0
                    or total_seconds + dur > comp_max_seconds
                    or i not in available
                ):
                    continue
                candidates.append(i)

            if not candidates:
                continue

            scores = []
            for cand in candidates:
                vid_id = video_ids[cand]
                dur = video_durations[vid_id]

                q = 1.0 / (D[a_idx, cand] + 1e-6)
                diversity_gain = (
                    min([D[cand, s["idx"]] for s in selected])
                    if selected
                    else float(np.max(D[a_idx]))
                )
                usage_penalty = 1.0 / (1.0 + usage_count[cand])
                stratum_bonus = 1.0 / (1.0 + stratum_counts[stratum])

                score = (
                    (q**theme_coherence_power)
                    * ((diversity_gain + 1e-6) ** diversity_power)
                    * usage_penalty
                    * stratum_bonus
                )
                scores.append(score)

            best = candidates[int(np.argmax(scores))]
            vid_id = video_ids[best]
            dur = video_durations[vid_id]

            selected.append({"idx": best, "secs": float(dur)})
            stratum_counts[stratum] += 1
            available.discard(best)
            total_seconds += dur

            if total_seconds >= comp_max_seconds:
                break

        # Fallback: if the minimum duration is not met, add remaining videos
        attempts = 0
        fallback_triggered = False
        while total_seconds < comp_min_seconds and available and attempts < len(Z_norm):
            if not fallback_triggered:
                print(
                    f"  WARNING: Compilation {ci+1}: semantic pool exhausted, using fallback to meet duration target"
                )
                fallback_triggered = True
            attempts += 1
            candidates = []
            for cand in list(available):
                vid_id = video_ids[cand]
                dur = video_durations.get(vid_id, 0.0)
                if dur <= 0 or total_seconds + dur > comp_max_seconds:
                    continue
                candidates.append(cand)

            if not candidates:
                break

            scores = []
            for cand in candidates:
                vid_id = video_ids[cand]
                dur = video_durations[vid_id]

                q = 1.0 / (D[a_idx, cand] + 1e-6)
                diversity_gain = (
                    min([D[cand, s["idx"]] for s in selected])
                    if selected
                    else float(np.max(D[a_idx]))
                )
                usage_penalty = 1.0 / (1.0 + usage_count[cand])
                s = stratum_labels[cand]
                stratum_bonus = 1.0 / (1.0 + stratum_counts[s])

                score = (
                    (q**theme_coherence_power)
                    * ((diversity_gain + 1e-6) ** diversity_power)
                    * usage_penalty
                    * stratum_bonus
                )
                scores.append(score)

            best = candidates[int(np.argmax(scores))]
            vid_id = video_ids[best]
            dur = video_durations[vid_id]

            selected.append({"idx": best, "secs": float(dur)})
            available.discard(best)
            s = stratum_labels[best]
            stratum_counts[s] += 1
            total_seconds += dur

            if total_seconds >= comp_max_seconds:
                break

        for s in selected:
            usage_count[s["idx"]] += 1

        compilations.append(
            {"anchor_idx": a_idx, "members": selected, "total_seconds": total_seconds}
        )

    return compilations


# ============================================================
# Frontend Interface
# ============================================================


def run(
    pipeline_context: Dict,
    n_compilations: int = 10,
    comp_min_seconds: float = 55.0,
    comp_max_seconds: float = 65.0,
    anchor_spread_method: str = "euclidean",
    theme_coherence_power: float = 1.0,
    diversity_power: float = 1.0,
    seed: int | None = None,
) -> Dict:
    """
    Step 6: Select compilations from strata.

    Selects N diverse compilations by:
    1. Choosing N diverse anchor videos using k-center algorithm
    2. For each anchor, greedily selecting videos to hit duration targets
    3. Balancing theme coherence, diversity, and stratum representation

    Parameters
    ----------
    pipeline_context : dict
        Pipeline state from previous step (contains stratum_labels, embedding_Z)
    n_compilations : int
        Number of compilations to create
    comp_min_seconds : float
        Minimum duration per compilation (seconds)
    comp_max_seconds : float
        Maximum duration per compilation (seconds)
    anchor_spread_method : str
        Distance metric for k-center: 'euclidean' or 'cosine'
    theme_coherence_power : float
        Weight for coherence with anchor in scoring (higher = more coherent)
    diversity_power : float
        Weight for diversity from existing selections (higher = more diverse)
    seed : int or None
        Random seed

    Returns
    -------
    dict
        Updated pipeline context with compilations and compilation_dir
    """
    if seed is not None:
        set_seeds(seed)

    output_dir = Path(pipeline_context["output_root"])
    embedding_Z = pipeline_context["embedding_Z"]
    stratum_labels = pipeline_context["stratum_labels"]
    video_ids = pipeline_context["video_ids"]
    processed_videos_dir = pipeline_context["processed_videos_dir"]

    # Get video durations
    from src.steps.step1a_normalize_videos import scan_videos
    from src.steps.step1b_extract_frames import get_video_duration

    video_durations = {}
    videos = scan_videos(Path(processed_videos_dir))
    vid_files = {Path(v).stem: Path(v).resolve() for v in videos}

    for vid in video_ids:
        p = vid_files.get(vid)
        if p is None:
            video_durations[vid] = 0.0
        else:
            d = get_video_duration(p)
            video_durations[vid] = float(d) if d is not None else 0.0

    # Select compilations
    compilations = select_compilations(
        embedding_Z,
        stratum_labels,
        video_ids,
        video_durations,
        n_compilations=n_compilations,
        comp_min_seconds=comp_min_seconds,
        comp_max_seconds=comp_max_seconds,
        anchor_spread_method=anchor_spread_method,
        theme_coherence_power=theme_coherence_power,
        diversity_power=diversity_power,
        seed=seed or 1234,
    )

    # Save metrics
    metrics = {
        "n_compilations": len(compilations),
        "n_strata": int(pipeline_context["n_strata"]),
        "compilation_details": [
            {
                "comp_id": int(ci),
                "anchor_idx": int(comp["anchor_idx"]),
                "n_members": int(len(comp["members"])),
                "total_seconds": float(comp["total_seconds"]),
            }
            for ci, comp in enumerate(compilations)
        ],
    }
    with open(output_dir / "selection_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    compilation_dir = ensure_dir(get_compilations_dir(output_dir))

    return {
        **pipeline_context,
        "compilations": compilations,
        "compilation_dir": compilation_dir,
        "video_files": vid_files,
    }


__all__ = [
    "run",
    "kcenter_greedy",
    "select_compilations",
]
