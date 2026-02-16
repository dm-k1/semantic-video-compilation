"""
Step 1b: Extract and save frames from normalized videos.

Contains all functionality for:
- Frame extraction from videos at specified FPS
- Building frames index (dataframe)
- Saving frame data to disk
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.core import (
    ensure_dir,
    get_frames_dir,
    run_ffmpeg,
    safe_delete_dir,
    ffprobe_json,
)
from src.data import validate_frames_df

# ============================================================
# Frame Extraction
# ============================================================


def get_video_duration(video_path: Path) -> float | None:
    """Return duration (seconds) for `video_path`, or None on failure."""
    try:
        data = ffprobe_json(video_path, show_entries="format=duration")
        fmt = data.get("format", {})
        if "duration" in fmt:
            return float(fmt["duration"])
    except Exception:
        return None
    return None


def ffmpeg_extract_frames(
    video_path: Path,
    out_dir: Path,
    fps: int = 3,
    start: float = 0.0,
    end: float | None = None,
    pattern: str = "{stem}_%06d.jpg",
) -> List[Path]:
    """Extract frames from `video_path` into `out_dir` and return list of saved paths."""
    out_dir = ensure_dir(out_dir)
    stem = Path(video_path).stem
    out_pattern = str(out_dir / pattern.format(stem=stem))

    cmd = ["ffmpeg", "-y"]
    if start and start > 0.0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video_path)]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-r", str(fps), out_pattern]

    try:
        run_ffmpeg(cmd)
    except Exception:
        return []

    return sorted(out_dir.glob(f"{stem}_*.jpg"))


def extract_and_save_frames(
    pipeline_context: Dict, fps: int = 3, overwrite: bool = True
) -> Dict:
    """Extract frames from normalized videos and save index to disk."""
    output_root = Path(pipeline_context["output_root"])
    videos_dir = Path(pipeline_context["processed_videos_dir"])
    frames_dir = ensure_dir(get_frames_dir(output_root))

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    # Scan for video files
    VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    video_paths = [
        str(x) for ext in VIDEO_EXTS for x in videos_dir.glob(f"*{ext}") if x.is_file()
    ]

    if not video_paths:
        raise FileNotFoundError(f"No videos found in {videos_dir}")

    frames_index: List[Dict] = []
    for v in video_paths:
        vid = Path(v).stem
        out_sub = frames_dir / vid

        if overwrite and out_sub.exists():
            safe_delete_dir(out_sub, root=output_root)

        dur = get_video_duration(Path(v))
        if dur is None or dur <= 0.0:
            continue

        files = ffmpeg_extract_frames(
            Path(v),
            out_sub,
            fps=fps,
            start=0.0,
            end=float(dur),
            pattern="{stem}_%06d.jpg",
        )
        for idx, p in enumerate(files):
            frames_index.append(
                {"video_id": vid, "frame_idx": idx, "frame_path": str(p)}
            )

    frames_df = pd.DataFrame(frames_index)
    validate_frames_df(frames_df)
    frames_csv_path = output_root / "frames_index.csv"
    frames_df.to_csv(frames_csv_path, index=False)

    return {
        **pipeline_context,
        "frames_df": frames_df,
        "frames_dir": frames_dir,
        "n_frames": len(frames_df),
    }


# ============================================================
# Frontend Interface
# ============================================================


def run(
    pipeline_context: Dict,
    fps: int = 3,
    overwrite_frames: bool = True,
    seed: int | None = None,
) -> Dict:
    """
    Step 1b: Extract and save frames from normalized videos.

    Parameters
    ----------
    pipeline_context : dict
        Pipeline state from step 1a (contains processed_videos_dir, output_root)
    fps : int
        Frame extraction rate (frames per second)
    overwrite_frames : bool
        If True, delete existing frames and re-extract
    seed : int or None
        Random seed (unused in this step but kept for consistency)

    Returns
    -------
    dict
        Updated pipeline context with frames_df and frames_dir
    """
    pipeline = extract_and_save_frames(
        pipeline_context, fps=fps, overwrite=overwrite_frames
    )

    return pipeline


__all__ = [
    "run",
    "get_video_duration",
    "ffmpeg_extract_frames",
    "extract_and_save_frames",
]
