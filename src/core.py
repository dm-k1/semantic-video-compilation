"""
Core utilities: configuration, paths, reporting, seeds, and shell helpers.
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None


# ------------------------------
# Config
# ------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    videos_dir: Path

    # Step 1: Video import & frame extraction
    sanitize_names: bool = True
    normalize_videos: bool = True
    target_resolution: Optional[int] = None
    uniform_resolution: bool = False
    force_reimport: bool = False
    fps: int = 3
    overwrite_frames: bool = True

    # Step 2: CLIP encoding
    clip_model_name: str = "ViT-B/16"
    scene_source: Optional[Union[str, Path, list, dict]] = None

    # Step 3: Aggregation
    aggregation_method: str = "percentile"
    percentile: int = 95

    # Step 4: Dimensionality reduction
    svd_variance_threshold: float = 0.99
    use_varimax: bool = True

    # Step 5: Clustering
    target_compilation_duration: float = 60.0
    min_videos_per_compilation: int = 3
    cluster_quality_threshold: float = 0.15
    max_k: int = 20

    # Step 6-7: Compilation selection & rendering
    n_compilations: int = 10
    comp_min_seconds: float = 55.0
    comp_max_seconds: float = 65.0

    # Global seed
    seed: int = 1234


# ------------------------------
# Paths
# ------------------------------


def get_output_root(videos_dir: Path) -> Path:
    return Path(videos_dir).parent / "output"


def get_processed_videos_dir(output_root: Path) -> Path:
    return Path(output_root) / "processed_videos"


def get_frames_dir(output_root: Path) -> Path:
    return Path(output_root) / "frames"


def get_compilations_dir(output_root: Path) -> Path:
    return Path(output_root) / "compilations"


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_delete_dir(path: Path, root: Path) -> None:
    """
    Safely delete directory inside root, avoid nuking wrong paths.
    """
    import shutil

    path_res = Path(path).resolve()
    root_res = Path(root).resolve()
    if not path_res.exists():
        return
    if path_res == root_res:
        raise ValueError(f"Refusing to delete root: {root_res}")
    if str(root_res) not in str(path_res):
        raise ValueError(f"Refusing to delete outside root: {path_res}")
    shutil.rmtree(path_res)


# ------------------------------
# Reporting
# ------------------------------


def summarize_import(pipeline_context: dict) -> None:
    if "processed_videos_dir" not in pipeline_context:
        print(
            "WARNING: Import step not completed. Summary unavailable."
        )
        return

    print("=" * 60)
    print("IMPORT SUMMARY")
    print("=" * 60)

    output_root = Path(pipeline_context["output_root"])
    processed_dir = Path(pipeline_context["processed_videos_dir"])
    n_videos = pipeline_context.get("n_videos", 0)

    print(f"\nOutput Directory: {output_root}")
    print(f"Videos Imported: {n_videos}")
    print(f"Processed Videos: {processed_dir}")

    metadata_path = output_root / "import_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        print("\nProcessing Settings:")
        print(f"   Sanitize names: {metadata.get('sanitize_names', 'N/A')}")
        print(f"   Normalize: {metadata.get('normalize', 'N/A')}")
        print(f"   Target resolution: {metadata.get('target_resolution', 'max')}")
        print(f"   Uniform resolution: {metadata.get('uniform_resolution', False)}")

    print()


def summarize_frames(pipeline_context: dict) -> None:
    if "frames_df" not in pipeline_context:
        print(
            "WARNING: Frame extraction not completed. Summary unavailable."
        )
        return

    print("=" * 60)
    print("FRAME EXTRACTION SUMMARY")
    print("=" * 60)

    frames_df = pipeline_context["frames_df"]
    n_frames = len(frames_df)
    n_videos = frames_df["video_id"].nunique()

    print(f"\nTotal Frames: {n_frames:,}")
    print(f"Videos Sampled: {n_videos}")
    print("Frames per Video:")

    frames_per_video = frames_df.groupby("video_id").size()
    print(f"   Mean: {frames_per_video.mean():.1f}")
    print(f"   Median: {frames_per_video.median():.1f}")
    print(f"   Range: [{frames_per_video.min()}, {frames_per_video.max()}]")

    print("\nDistribution:")
    quantiles = frames_per_video.quantile([0.25, 0.5, 0.75])
    print(f"   25th percentile: {quantiles[0.25]:.0f} frames")
    print(f"   50th percentile: {quantiles[0.5]:.0f} frames")
    print(f"   75th percentile: {quantiles[0.75]:.0f} frames")

    print()


def summarize_encoding(pipeline_context: dict, top_n: int = 5) -> None:
    if "frame_scores" not in pipeline_context:
        print(
            "WARNING: CLIP encoding not completed. Summary unavailable."
        )
        return

    print("=" * 60)
    print("CLIP ENCODING SUMMARY")
    print("=" * 60)

    frame_scores = pipeline_context["frame_scores"]
    n_frames, n_scenes = frame_scores.shape
    n_encoded = pipeline_context.get("n_frames_encoded", n_frames)

    print(f"\nModel: {pipeline_context.get('clip_model', 'ViT-B/16')}")
    print(f"Frames Encoded: {n_encoded:,} / {n_frames:,}")
    print(f"Scene Statements: {n_scenes}")

    print("\nScore Statistics:")
    print(f"   Mean: {frame_scores.mean():.4f}")
    print(f"   Std: {frame_scores.std():.4f}")
    print(f"   Range: [{frame_scores.min():.4f}, {frame_scores.max():.4f}]")

    max_scores = frame_scores.max(axis=1)
    top_indices = np.argsort(max_scores)[-top_n:][::-1]

    print(f"\nTop {top_n} Scoring Frames (max across all scenes):")
    frames_df = pipeline_context.get("frames_df")
    if frames_df is not None:
        for rank, idx in enumerate(top_indices, 1):
            score = max_scores[idx]
            video_id = (
                frames_df.iloc[idx]["video_id"] if idx < len(frames_df) else "unknown"
            )
            frame_idx = (
                frames_df.iloc[idx]["frame_idx"] if idx < len(frames_df) else idx
            )
            print(
                f"   {rank}. Video: {video_id}, Frame: {frame_idx}, Score: {score:.4f}"
            )

    print("\nScene Coverage (mean score per scene):")
    scene_means = frame_scores.mean(axis=0)
    top_scenes = np.argsort(scene_means)[-top_n:][::-1]

    scene_statements = pipeline_context.get("scene_statements", [])
    for rank, scene_idx in enumerate(top_scenes, 1):
        score = scene_means[scene_idx]
        scene_text = (
            scene_statements[scene_idx]
            if scene_idx < len(scene_statements)
            else f"Scene {scene_idx}"
        )
        print(f"   {rank}. {scene_text[:50]}... Score: {score:.4f}")

    print()


def summarize_aggregation(pipeline_context: dict, top_n: int = 5) -> None:
    if "video_scores" not in pipeline_context:
        print("WARNING: Video aggregation not completed. Summary unavailable.")
        return

    print("=" * 60)
    print("VIDEO AGGREGATION SUMMARY")
    print("=" * 60)

    video_scores = pipeline_context["video_scores"]
    video_ids = pipeline_context.get("video_ids", [])
    n_videos, n_scenes = video_scores.shape

    print(f"\nVideos: {n_videos}")
    print(f"Scene Dimensions: {n_scenes}")
    method = pipeline_context.get("aggregation_method", "percentile")
    if method == "percentile":
        print(
            f"Aggregation Method: {pipeline_context.get('percentile', 95)}th percentile"
        )
    else:
        print(f"Aggregation Method: {method}")

    print("\nScore Distribution:")
    print(f"   Mean: {video_scores.mean():.4f}")
    print(f"   Std: {video_scores.std():.4f}")
    print(f"   Range: [{video_scores.min():.4f}, {video_scores.max():.4f}]")

    max_scores = video_scores.max(axis=1)
    top_indices = np.argsort(max_scores)[-top_n:][::-1]

    print(f"\nTop {top_n} Videos (by max scene score):")
    for rank, idx in enumerate(top_indices, 1):
        video_id = video_ids[idx] if idx < len(video_ids) else f"video_{idx}"
        score = max_scores[idx]
        best_scene = np.argmax(video_scores[idx])
        print(f"   {rank}. {video_id}: {score:.4f} (best: scene {best_scene})")

    print("\nScene Preference (videos per scene as top match):")
    best_scenes = np.argmax(video_scores, axis=1)
    scene_counts = pd.Series(best_scenes).value_counts().head(top_n)
    for scene_idx, count in scene_counts.items():
        print(f"   Scene {scene_idx}: {count} videos")

    print()


def summarize_svd(pipeline_context: dict) -> None:
    if "svd_scores" not in pipeline_context:
        print("WARNING: SVD reduction not completed. Summary unavailable.")
        return

    print("=" * 60)
    print("DIMENSIONALITY REDUCTION SUMMARY")
    print("=" * 60)

    svd_scores = pipeline_context["svd_scores"]
    video_scores = pipeline_context.get("video_scores")
    n_videos, n_components = svd_scores.shape

    original_dims = video_scores.shape[1] if video_scores is not None else "unknown"

    print(f"\nDimension Reduction: {original_dims} → {n_components}")
    print(f"Videos: {n_videos}")
    print(
        f"Variance Threshold: {pipeline_context.get('variance_threshold', 0.99)*100:.1f}%"
    )

    print("\nFeature Distribution:")
    print(f"   Mean: {svd_scores.mean():.4f}")
    print(f"   Std: {svd_scores.std():.4f}")
    print(f"   Range: [{svd_scores.min():.4f}, {svd_scores.max():.4f}]")

    print("\nPer-Component Statistics:")
    for i in range(min(5, n_components)):
        comp_scores = svd_scores[:, i]
        print(
            f"   Component {i}: mean={comp_scores.mean():.3f}, std={comp_scores.std():.3f}"
        )

    print()


def summarize_strata(pipeline_context: dict) -> None:
    if "stratum_labels" not in pipeline_context:
        print("WARNING: Strata not computed. Summary unavailable.")
        return

    print("=" * 60)
    print("STRATA SUMMARY")
    print("=" * 60)

    stratum_labels = pipeline_context["stratum_labels"]
    n_strata = len(set(stratum_labels))
    n_videos = len(stratum_labels)

    print(f"\nNumber of Strata: {n_strata}")
    print(f"Videos: {n_videos}")

    sizes = pd.Series(stratum_labels).value_counts().sort_index()
    print("\nStratum Sizes:")
    for stratum, count in sizes.items():
        print(f"   Stratum {stratum}: {count} videos")

    print()


def summarize_compilations(pipeline_context: dict) -> None:
    if "compilations" not in pipeline_context:
        print(
            "WARNING: Compilations not generated. Summary unavailable."
        )
        return

    print("=" * 60)
    print("COMPILATION SUMMARY")
    print("=" * 60)

    comps = pipeline_context["compilations"]
    n_comp = len(comps)

    print(f"\nNumber of Compilations: {n_comp}")

    total_durations = [c["total_seconds"] for c in comps]
    print(
        f"Duration: mean={np.mean(total_durations):.1f}s, "
        f"min={np.min(total_durations):.1f}s, max={np.max(total_durations):.1f}s"
    )

    print()


def summary(pipeline_context: dict, step: str) -> None:
    """Unified summary interface."""
    step = step.lower()
    if step in ("import", "step1", "step1a"):
        summarize_import(pipeline_context)
    elif step in ("frames", "step1b"):
        summarize_frames(pipeline_context)
    elif step in ("encoding", "step2"):
        summarize_encoding(pipeline_context)
    elif step in ("aggregate", "step3"):
        summarize_aggregation(pipeline_context)
    elif step in ("svd", "reduce", "step4"):
        summarize_svd(pipeline_context)
    elif step in ("strata", "step5"):
        summarize_strata(pipeline_context)
    elif step in ("compile", "compilations", "step6"):
        summarize_compilations(pipeline_context)
    else:
        print(f"WARNING: Unknown summary step: {step}")


# ------------------------------
# Seeds
# ------------------------------


def set_seeds(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ------------------------------
# Shell / FFmpeg helpers
# ------------------------------


def run_ffmpeg(cmd: List[str]) -> None:
    """Run ffmpeg command and raise error if fails."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip().split("\n")[-5:]
        raise RuntimeError("FFmpeg failed: " + "\n".join(err))


def ffprobe_json(
    video_path: Path, show_entries: str, select_streams: str = ""
) -> Dict[str, Any]:
    """Return ffprobe info in JSON format."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        show_entries,
    ]
    if select_streams:
        cmd += ["-select_streams", select_streams]
    cmd += [str(video_path)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip().split("\n")[-5:]
        raise RuntimeError("ffprobe failed: " + "\n".join(err))

    return json.loads(proc.stdout)


__all__ = [
    "PipelineConfig",
    "get_output_root",
    "get_processed_videos_dir",
    "get_frames_dir",
    "get_compilations_dir",
    "ensure_dir",
    "safe_delete_dir",
    "summarize_import",
    "summarize_frames",
    "summarize_encoding",
    "summarize_aggregation",
    "summarize_svd",
    "summarize_strata",
    "summarize_compilations",
    "summary",
    "set_seeds",
    "run_ffmpeg",
    "ffprobe_json",
]
