"""
Step 1a: Normalize videos (sanitize names and encode).

Contains all functionality for:
- Video scanning and file handling
- Filename sanitization for FFmpeg compatibility
- Video codec/resolution normalization
- Organized output structure setup
"""

from __future__ import annotations

import json
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.core import (
    ensure_dir,
    get_output_root,
    get_processed_videos_dir,
    run_ffmpeg,
    safe_delete_dir,
    ffprobe_json,
)

# ============================================================
# Video Scanning & Probing
# ============================================================

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def sanitize_name(name: str) -> str:
    """Convert filename into an FFmpeg-safe ASCII-only name."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    for ch in ["'", '"', "´", "`", "'", ","]:
        name = name.replace(ch, "")
    name = name.replace(" ", "_")
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def scan_videos(videos_dir: Path, exts: Iterable[str] = VIDEO_EXTS) -> List[str]:
    """Scan directory for video files and return list of paths (as strings)."""
    p = Path(videos_dir)
    return [str(x) for ext in exts for x in p.glob(f"*{ext}") if x.is_file()]


def get_video_resolution(video_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Get video resolution using ffprobe."""
    try:
        data = ffprobe_json(
            video_path, show_entries="stream=width,height", select_streams="v:0"
        )
        streams = data.get("streams", [])
        if streams:
            stream = streams[0]
            return stream.get("width"), stream.get("height")
    except Exception:
        return None, None
    return None, None


def get_video_duration(video_path: Path) -> Optional[float]:
    """Return duration (seconds) for `video_path`, or None on failure."""
    try:
        data = ffprobe_json(video_path, show_entries="format=duration")
        fmt = data.get("format", {})
        if "duration" in fmt:
            return float(fmt["duration"])
    except Exception:
        return None
    return None


# ============================================================
# Video Normalization
# ============================================================


def normalize_videos(
    videos_dir: Path,
    sanitize_names: bool = True,
    normalize: bool = False,
    target_resolution: Optional[int] = None,
    uniform_resolution: bool = False,
    force_reimport: bool = False,
    output_root: Optional[Path] = None,
) -> Dict:
    """
    Import and normalize videos from videos directory into organized output structure.
    Creates 'output' folder as sibling to videos folder, with subdirectories inside.
    """
    videos_dir = Path(videos_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    if output_root is None:
        output_root = ensure_dir(get_output_root(videos_dir))
    else:
        output_root = ensure_dir(Path(output_root))

    processed_videos_dir = ensure_dir(get_processed_videos_dir(output_root))

    if processed_videos_dir.exists():
        if force_reimport:
            safe_delete_dir(processed_videos_dir, root=output_root)
            processed_videos_dir = ensure_dir(get_processed_videos_dir(output_root))
        else:
            existing_videos = scan_videos(processed_videos_dir)
            if existing_videos:
                return {
                    "output_root": output_root,
                    "processed_videos_dir": processed_videos_dir,
                    "n_videos": len(existing_videos),
                    "video_paths": existing_videos,
                    "metadata": {"reused": True},
                }

    source_videos = scan_videos(videos_dir)
    if not source_videos:
        raise FileNotFoundError(f"No videos found in {videos_dir}")

    target_height: Optional[int] = None
    if normalize and uniform_resolution:
        if target_resolution:
            target_height = target_resolution
        else:
            max_height = 0
            for src in source_videos:
                _, h = get_video_resolution(Path(src))
                if h and h > max_height:
                    max_height = h
            target_height = max_height

    processed_videos: List[str] = []
    metadata = {
        "videos_dir": str(videos_dir),
        "output_root": str(output_root),
        "processed_videos_dir": str(processed_videos_dir),
        "sanitize_names": sanitize_names,
        "normalize": normalize,
        "target_resolution": target_resolution,
        "uniform_resolution": uniform_resolution,
        "videos": [],
    }

    for src_path in source_videos:
        src = Path(src_path)
        safe_stem = sanitize_name(src.stem) if sanitize_names else src.stem
        dst = processed_videos_dir / f"{safe_stem}.mp4"

        if normalize:
            _, src_height = get_video_resolution(src)
            if uniform_resolution and target_height:
                encode_height = target_height
            elif target_resolution and target_resolution < (src_height or 99999):
                encode_height = target_resolution
            else:
                encode_height = src_height

            cmd = [
                "ffmpeg",
                "-y",
                "-fflags",
                "+genpts",
                "-i",
                str(src),
            ]

            vf_filters: List[str] = []
            if encode_height and encode_height != src_height:
                vf_filters.append(f"scale=-2:{encode_height}")
            if vf_filters:
                cmd.extend(["-vf", ",".join(vf_filters)])

            cmd.extend(
                [
                    "-pix_fmt",
                    "yuv420p",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-movflags",
                    "+faststart",
                    str(dst),
                ]
            )

            try:
                run_ffmpeg(cmd)
                processed_videos.append(str(dst))
                metadata["videos"].append(
                    {
                        "source": str(src),
                        "output": str(dst),
                        "resolution": (
                            f"{encode_height}p" if encode_height else "original"
                        ),
                    }
                )
            except Exception as e:
                raise RuntimeError(f"FFmpeg failed while normalizing {src}: {e}") from e
        else:
            shutil.copy2(src, dst)
            processed_videos.append(str(dst))
            metadata["videos"].append(
                {"source": str(src), "output": str(dst), "resolution": "original"}
            )

    with open(output_root / "import_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "output_root": output_root,
        "processed_videos_dir": processed_videos_dir,
        "n_videos": len(processed_videos),
        "video_paths": processed_videos,
        "metadata": metadata,
    }


# ============================================================
# Frontend Interface
# ============================================================


def run(
    videos_dir,
    sanitize_names: bool = True,
    normalize: bool = True,
    target_resolution: int | None = None,
    uniform_resolution: bool = False,
    force_reimport: bool = False,
    seed: int | None = None,
    output_root: Path | None = None,
) -> Dict:
    """
    Step 1a: Normalize videos (sanitization and encoding).

    Parameters
    ----------
    videos_dir : str or Path
        Directory containing source videos
    sanitize_names : bool
        Clean filenames to ASCII-safe names
    normalize : bool
        Re-encode videos to h264/aac
    target_resolution : int or None
        Target resolution (720, 1080, etc.) or None for original
    uniform_resolution : bool
        If True, resize all videos to same resolution
    force_reimport : bool
        If True, delete existing output and reimport from scratch
    seed : int or None
        Random seed (unused in this step but kept for consistency)
    output_root : Path or None
        Override the default output root directory

    Returns
    -------
    dict
        Pipeline context with video metadata and paths
    """
    pipeline = normalize_videos(
        videos_dir=videos_dir,
        sanitize_names=sanitize_names,
        normalize=normalize,
        target_resolution=target_resolution,
        uniform_resolution=uniform_resolution,
        force_reimport=force_reimport,
        output_root=output_root,
    )

    return pipeline


__all__ = [
    "run",
    "sanitize_name",
    "scan_videos",
    "get_video_duration",
    "get_video_resolution",
    "normalize_videos",
]
