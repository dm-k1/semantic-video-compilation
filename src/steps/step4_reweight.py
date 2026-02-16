"""
Step 4: Representation preparation (centering + SVD-informed reweighting).

Contains all functionality for:
- Centering video score matrix
- Estimating signal subspace dimension via TruncatedSVD
- Computing statement weights from communalities
- Applying smooth weights to produce final video_scores
- Saving diagnostics and reports
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
# Helper Function 1: Centering
# ============================================================


def center_scores(X: np.ndarray, mode: str = "col") -> np.ndarray:
    """
    Center score matrix using various centering strategies.

    Parameters
    ----------
    X : np.ndarray
        Score matrix (n_videos, n_statements)
    mode : str
        Centering mode:
        - 'none': No centering (return X unchanged)
        - 'col': Column-wise centering (subtract mean of each statement across videos)
        - 'row': Row-wise centering (subtract mean of each video across statements)
        - 'both': Double-centering (remove row means, column means, add grand mean)

    Returns
    -------
    np.ndarray
        Centered matrix with same shape as X
    """
    if mode == "none":
        return X.copy()

    if mode == "col":
        col_mean = X.mean(axis=0, keepdims=True)
        return X - col_mean

    if mode == "row":
        row_mean = X.mean(axis=1, keepdims=True)
        return X - row_mean

    if mode == "both":
        row_mean = X.mean(axis=1, keepdims=True)
        col_mean = X.mean(axis=0, keepdims=True)
        grand_mean = X.mean()
        return X - row_mean - col_mean + grand_mean

    raise ValueError(
        "Unknown centering mode: {mode}. Must be one of: 'none', 'col', 'row', 'both'".format(
            mode=mode
        )
    )


# ============================================================
# Helper Function 2: SVD-based weights
# ============================================================


def compute_statement_weights(
    Xc: np.ndarray, variance_threshold: float = 0.95, k_cap: int = 32
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Compute statement weights using SVD communalities.

    Returns
    -------
    weights : np.ndarray
        Statement weights (n_statements,)
    explained_variance_ratio : np.ndarray
        Explained variance ratio for components up to max_components
    k95 : int
        Components needed to reach variance_threshold
    k_selected : int
        Selected component count after applying k_cap
    max_components : int
        Max components used for SVD fit
    """
    n_videos, n_statements = Xc.shape
    max_components = min(n_videos - 1, n_statements - 1, 128)
    max_components = max(1, max_components)

    svd = TruncatedSVD(n_components=max_components, random_state=0)
    svd.fit(Xc)

    evr = svd.explained_variance_ratio_
    cum = np.cumsum(evr)
    k95 = int(np.searchsorted(cum, variance_threshold) + 1)
    k95 = max(1, min(k95, max_components))

    k_selected = min(k95, k_cap, max_components)
    k_selected = max(1, k_selected)

    components = svd.components_[:k_selected, :]
    explained_variance = svd.explained_variance_[:k_selected]

    # PCA-style loadings: (n_statements, k)
    loadings = components.T * np.sqrt(explained_variance)[None, :]
    communalities = np.sum(loadings**2, axis=1)

    return communalities, evr, k95, k_selected, max_components


# ============================================================
# Frontend Interface
# ============================================================


def run(
    pipeline_context: Dict,
    centering: str = "col",
    variance_threshold: float = 0.95,
    k_cap: int = 32,
    beta: float = 0.5,
    seed: int | None = None,
) -> Dict:
    """
    Step 4: Center and reweight video scores using SVD-informed communalities.

    Parameters
    ----------
    pipeline_context : dict
        Pipeline state from previous step (contains video_scores_raw)
    centering : str
        Centering mode applied to video_scores: 'none', 'col', 'row', 'both'
    variance_threshold : float
        Cumulative variance threshold to select k95
    k_cap : int
        Maximum number of components to retain
    beta : float
        Weight strength (0,1], applied as w = sqrt(communalities)^beta
    seed : int or None
        Random seed

    Returns
    -------
    dict
        Updated pipeline context with centered and reweighted scores
    """
    if seed is not None:
        set_seeds(seed)

    output_dir = Path(pipeline_context["output_root"])
    X_raw = pipeline_context["video_scores_raw"]

    # Centering
    Xc = center_scores(X_raw, mode=centering)

    # Compute communalities and k selection
    communalities, evr, k95, k_selected, max_components = compute_statement_weights(
        Xc, variance_threshold=variance_threshold, k_cap=k_cap
    )

    eps = 1e-8
    w_raw = np.sqrt(communalities + eps)
    w = (w_raw) ** beta
    w = w / (w.mean() + eps)

    X_out = (Xc * w[None, :]).astype(np.float32)
    Xc_out = Xc.astype(np.float32)
    w_out = w.astype(np.float32)

    # Save artifacts
    np.save(output_dir / "video_scores_centered.npy", Xc_out)
    np.save(output_dir / "statement_weights.npy", w_out)
    np.save(output_dir / "video_scores.npy", X_out)
    np.save(output_dir / "svd_explained_variance_ratio.npy", evr.astype(np.float32))

    weights_df = pd.DataFrame(
        {
            "statement_index": np.arange(len(w_out)),
            "weight": w_out,
            "communality": communalities.astype(np.float32),
        }
    )
    weights_df.to_csv(output_dir / "statement_weights.csv", index=False)

    report = {
        "centering": centering,
        "variance_threshold": float(variance_threshold),
        "k95": int(k95),
        "k_selected": int(k_selected),
        "k_cap": int(k_cap),
        "max_components": int(max_components),
        "beta": float(beta),
        "n_videos": int(X_raw.shape[0]),
        "n_statements": int(X_raw.shape[1]),
    }
    with open(output_dir / "reweight_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return {
        **pipeline_context,
        "video_scores": X_out,
        "video_scores_centered": Xc_out,
        "statement_weights": w_out,
        "k_selected": k_selected,
        "k95": k95,
        "variance_threshold": variance_threshold,
        "beta": beta,
        "centering": centering,
    }


__all__ = ["run", "center_scores", "compute_statement_weights"]
