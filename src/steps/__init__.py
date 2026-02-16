"""
Step modules for the video compilation pipeline.

Each step encapsulates a complete stage of the pipeline:
- step1a_normalize_videos: Sanitize and normalize video files
- step1b_extract_frames: Extract frames from normalized videos
- step2_encode_score: Encode frames with CLIP and score scenes
- step3_aggregate: Aggregate frame scores to video level (raw)
- step4_reweight: Center and reweight scores with SVD-informed weights
- step5_make_strata: Create prototype-based strata for grouping
- step6_select_compilations: Select compilations from strata

Each step module exports a `run()` function that takes a pipeline context
and returns an updated pipeline context.
"""

from . import (
    step1a_normalize_videos,
    step1b_extract_frames,
    step2_encode_score,
    step3_aggregate,
    step4_reweight,
    step5_make_strata,
    step6_select_compilations,
)

__all__ = [
    "step1a_normalize_videos",
    "step1b_extract_frames",
    "step2_encode_score",
    "step3_aggregate",
    "step4_reweight",
    "step5_make_strata",
    "step6_select_compilations",
]
