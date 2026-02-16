"""
Data schemas and scene loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd

# ------------------------------
# Schemas
# ------------------------------

FRAME_COLS = ("video_id", "frame_idx", "frame_path")
VIDEO_INDEX_COLS = ("video_id",)
CLUSTER_COLS = ("video_id", "cluster")


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_frames_df(df: pd.DataFrame) -> None:
    require_columns(df, FRAME_COLS, "frames_df")


def validate_video_index_df(df: pd.DataFrame) -> None:
    require_columns(df, VIDEO_INDEX_COLS, "video_index_df")


def validate_cluster_df(df: pd.DataFrame) -> None:
    require_columns(df, CLUSTER_COLS, "video_clusters_df")


# ------------------------------
# Scenes
# ------------------------------


def load_scenes_from_csv(csv_path: Union[str, Path]) -> Dict[str, List[str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Scene CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "category" not in df.columns or "statement" not in df.columns:
        raise ValueError("CSV must have 'category' and 'statement' columns")

    scenes_dict: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        category = row["category"]
        statement = row["statement"]
        scenes_dict.setdefault(category, []).append(statement)

    print(
        f"OK: Loaded {len(df)} statements across {len(scenes_dict)} categories from {csv_path}"
    )
    return scenes_dict


def load_scenes_from_list(
    statements: List[str], default_category: str = "custom"
) -> Dict[str, List[str]]:
    if not isinstance(statements, list):
        raise ValueError("statements must be a list")

    scenes_dict = {default_category: statements}
    print(f"OK: Loaded {len(statements)} statements as category '{default_category}'")
    return scenes_dict


def get_scenes(
    source: Optional[Union[str, Path, list, dict]] = None,
    category: Optional[str] = None,
) -> List[str]:
    if source is None:
        raise ValueError("Scene source is required. Provide a CSV path, list, or dict.")
    if isinstance(source, (str, Path)):
        scenes_dict = load_scenes_from_csv(Path(source))
    elif isinstance(source, list):
        scenes_dict = load_scenes_from_list(source)
    elif isinstance(source, dict):
        scenes_dict = source
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    if category is not None:
        if category not in scenes_dict:
            raise ValueError(
                f"Category '{category}' not found. Available: {list(scenes_dict.keys())}"
            )
        scenes_dict = {category: scenes_dict[category]}

    all_scenes: List[str] = []
    for statements in scenes_dict.values():
        all_scenes.extend(statements)
    return all_scenes


__all__ = [
    "FRAME_COLS",
    "VIDEO_INDEX_COLS",
    "CLUSTER_COLS",
    "require_columns",
    "validate_frames_df",
    "validate_video_index_df",
    "validate_cluster_df",
    "load_scenes_from_csv",
    "load_scenes_from_list",
    "get_scenes",
]
