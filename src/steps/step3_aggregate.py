"""
Step 3: Aggregate frame scores to video level.

Contains all functionality for:
- Aggregating per-frame CLIP scores to per-video scores
- Supporting multiple aggregation methods (percentile, mean, median, max, min)
- Saving aggregated scores and video index
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.core import set_seeds
from src.data import validate_frames_df, validate_video_index_df

# ============================================================
# Helper Function 1: Aggregation (frames -> videos)
# ============================================================


def aggregate_video_scores(
    frames_df: pd.DataFrame,
    frame_scores: np.ndarray,
    method: str = "percentile",
    percentile: int = 95,
) -> Tuple[np.ndarray, list]:
    """Aggregate frame-level CLIP scores to video level."""
    video_ids = sorted(frames_df["video_id"].unique().tolist())
    n_videos = len(video_ids)
    n_text = frame_scores.shape[1]

    video_scores = np.zeros((n_videos, n_text), dtype=np.float32)

    for vi, vid in enumerate(video_ids):
        frame_mask = (frames_df["video_id"] == vid).values
        vid_frame_scores = frame_scores[frame_mask]
        if vid_frame_scores.size == 0:
            continue

        if method == "percentile":
            video_scores[vi, :] = np.percentile(vid_frame_scores, percentile, axis=0)
        elif method == "mean":
            video_scores[vi, :] = np.mean(vid_frame_scores, axis=0)
        elif method == "median":
            video_scores[vi, :] = np.median(vid_frame_scores, axis=0)
        elif method == "max":
            video_scores[vi, :] = np.max(vid_frame_scores, axis=0)
        elif method == "min":
            video_scores[vi, :] = np.min(vid_frame_scores, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return video_scores, video_ids


# ============================================================
# Frontend Interface
# ============================================================


def run(
    pipeline_context: Dict,
    method: str = "percentile",
    percentile: int = 95,
    seed: int | None = None,
) -> Dict:
    """
    Step 3: Aggregate frame scores to video level.

    Parameters
    ----------
    pipeline_context : dict
        Pipeline state from previous step (contains frame_scores)
    method : str
        Aggregation method: 'percentile', 'mean', 'median', 'max', 'min'
    percentile : int
        Percentile to use if method='percentile' (default 95)
    seed : int or None
        Random seed

    Returns
    -------
    dict
        Updated pipeline context with video_scores_raw and video_ids
    """
    if seed is not None:
        set_seeds(seed)

    output_dir = Path(pipeline_context["output_root"])
    frames_df = pipeline_context["frames_df"]
    frame_scores = pipeline_context["frame_scores"]

    validate_frames_df(frames_df)

    # Aggregate scores (frames -> videos)
    video_scores_raw, video_ids = aggregate_video_scores(
        frames_df, frame_scores, method=method, percentile=percentile
    )

    video_scores_raw = video_scores_raw.astype(np.float32)

    # Save results
    np.save(output_dir / "video_scores_raw.npy", video_scores_raw)
    video_index_df = pd.DataFrame({"video_id": video_ids})
    validate_video_index_df(video_index_df)
    video_index_df.to_csv(output_dir / "video_index.csv", index=False)

    return {
        **pipeline_context,
        "video_scores_raw": video_scores_raw,
        "video_ids": video_ids,
        "video_index_df": video_index_df,
        "aggregation_method": method,
        "percentile": percentile,
    }


__all__ = [
    "run",
    "aggregate_video_scores",
]
