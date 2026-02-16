"""
Step 2: Encode frames with CLIP and score scenes.

Contains all functionality for:
- Loading CLIP model
- Encoding scene text statements
- Computing frame-level similarity scores
- Saving encoded features to disk
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.core import set_seeds
from src.data import get_scenes, validate_frames_df

try:
    import torch
except Exception:
    torch = None

try:
    import clip
except Exception:
    clip = None


# ============================================================
# CLIP Model Utilities
# ============================================================


def get_torch_device():
    """Get appropriate torch device (CUDA if available, else CPU)."""
    if torch is None:
        return None
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip_model(model_name: str):
    """Load CLIP model and preprocessing function."""
    if clip is None or torch is None:
        raise RuntimeError("CLIP model not available. Install torch and clip.")
    device = get_torch_device()
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device


# ============================================================
# Text & Frame Encoding
# ============================================================


def encode_text_features(scene_statements: List[str], model, device) -> np.ndarray:
    """Encode scene text statements to CLIP text features."""
    tokens = [clip.tokenize(s) for s in scene_statements]
    text_tokens = torch.cat(tokens).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats.cpu().numpy()


def score_frames(
    frames_df, text_features: np.ndarray, model, preprocess, device
) -> Tuple[np.ndarray, int]:
    """Score all frames against all text features using CLIP."""
    n_frames = len(frames_df)
    n_text = text_features.shape[0]
    frame_scores = np.zeros((n_frames, n_text), dtype=np.float32)

    model.eval()
    valid_count = 0
    with torch.no_grad():
        text_t = torch.from_numpy(text_features).to(device)
        for i, row in enumerate(frames_df.itertuples(index=False)):
            img_path = row.frame_path
            try:
                img = Image.open(img_path).convert("RGB")
                img_t = preprocess(img).unsqueeze(0).to(device)
                img_feat = model.encode_image(img_t)
                img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-12)
                sims = (img_feat @ text_t.T).cpu().numpy().reshape(-1)
                frame_scores[i, :] = sims
                valid_count += 1
            except Exception:
                continue

    return frame_scores, valid_count


# ============================================================
# Frontend Interface
# ============================================================


def run(
    pipeline_context: Dict,
    scene_statements: List[str] | None = None,
    scene_source: str | None = None,
    model_name: str = "ViT-B/16",
    seed: int | None = None,
) -> Dict:
    """
    Step 2: Encode frames with CLIP and score scenes.

    Parameters
    ----------
    pipeline_context : dict
        Pipeline state from previous step
    scene_statements : list of str, optional
        Explicit list of scene statements to score
    scene_source : str, optional
        Path to CSV file with scene statements, or None for built-in scenes
    model_name : str
        CLIP model identifier (e.g., 'ViT-B/16', 'ViT-L/14')
    seed : int or None
        Random seed

    Returns
    -------
    dict
        Updated pipeline context with frame scores and text features
    """
    if seed is not None:
        set_seeds(seed)

    output_dir = Path(pipeline_context["output_root"])
    frames_df = pipeline_context["frames_df"]
    validate_frames_df(frames_df)

    # Get scene statements if not provided
    if scene_statements is None:
        scene_statements = get_scenes(source=scene_source)
        print(f"Loaded {len(scene_statements)} scene statements")

    # Load model and encode text
    model, preprocess, device = load_clip_model(model_name)
    text_features_np = encode_text_features(scene_statements, model, device)

    # Score frames
    frame_scores, valid_count = score_frames(
        frames_df, text_features_np, model, preprocess, device
    )

    # Save results
    np.savez_compressed(output_dir / "frame_scores.npz", frame_scores=frame_scores)

    return {
        **pipeline_context,
        "frame_scores": frame_scores,
        "text_features": text_features_np,
        "n_frames_encoded": valid_count,
        "scene_statements": scene_statements,
        "clip_model": model_name,
    }


__all__ = [
    "run",
    "get_torch_device",
    "load_clip_model",
    "encode_text_features",
    "score_frames",
]
