# Steps Directory

This directory contains individual step modules that each encapsulate a complete stage of the video compilation pipeline.

## Structure

Each step is a separate Python module that exports a `run()` function. This structure supports:
- **Modularity**: Each step is independent and self-contained
- **Clarity**: Each module's docstring describes its scope
- **Reusability**: Steps can be imported and called independently
- **Testing**: Each step can be tested in isolation

## Steps

### step1a_normalize_videos.py
**Purpose**: Import and normalize videos

Handles:
- Importing videos from a source directory
- Sanitizing filenames for FFmpeg
- Optional codec/resolution normalization
- Organizing output into a processed videos directory

**Input**: Videos directory path and configuration  
**Output**: Pipeline context with video metadata

### step1b_extract_frames.py
**Purpose**: Extract frames from normalized videos

Handles:
- Frame extraction at a specified FPS rate
- Building a frames index (CSV + DataFrame)
- Saving frame data to disk

**Input**: Pipeline context from step 1a  
**Output**: Pipeline context with frames information

### step2_encode_score.py
**Purpose**: Encode frames with CLIP and score scenes

Handles:
- Loading CLIP model
- Encoding scene text statements
- Computing frame-level similarity scores for each scene
- Saving encoded features to disk

**Input**: Pipeline context from step 1b  
**Output**: Pipeline context with frame scores and text features

### step3_aggregate.py
**Purpose**: Aggregate frame scores to video level

Handles:
- Aggregating per-frame CLIP scores to per-video scores
- Supporting multiple aggregation methods (percentile, mean, median, max, min)
- Saving aggregated scores and video index

**Input**: Pipeline context with frame scores  
**Output**: Pipeline context with video-level scores

### step4_reweight.py
**Purpose**: Center and reweight video scores

Handles:
- Centering video score matrix
- Estimating signal subspace dimension via TruncatedSVD
- Computing statement weights from communalities
- Applying smooth weights to produce final video scores
- Saving diagnostics and reports

**Input**: Pipeline context with video scores  
**Output**: Pipeline context with reweighted scores

### step5_make_strata.py
**Purpose**: Create prototype-based strata

Handles:
- Building low-dimensional SVD embedding of video scores
- Selecting K prototypes via farthest-first traversal
- Assigning videos to nearest prototype buckets
- Producing strata for compilation selection
- Automatic selection of strata count

**Input**: Pipeline context with reweighted scores  
**Output**: Pipeline context with strata assignments

### step6_select_compilations.py
**Purpose**: Select compilations from strata

Handles:
- Using k-center algorithm to select diverse anchor videos
- Greedily building compilations around each anchor
- Respecting duration constraints per compilation
- Balancing theme coherence, diversity, and stratum representation
- Saving compilation metadata and selection metrics

**Input**: Pipeline context with strata  
**Output**: Pipeline context with compilations and selection metrics

## Usage

### In the Notebook

The notebook shows imports from `src.steps` to convey the analysis flow:

```python
from src.steps.step1a_normalize_videos import run as step1a_normalize
from src.steps.step1b_extract_frames import run as step1b_extract_frames
```

### Direct Import of Step Modules

Direct imports are shown for conceptual reference:

```python
from src.steps import step1a_normalize_videos

result = step1a_normalize_videos.run(
    videos_dir="/path/to/videos",
    # ... other parameters
)
```

## Architecture

Each step module:
1. Imports only what it needs from core modules (analysis, video, data, etc.)
2. Defines a `run()` function with clear parameters and docstring
3. Calls the appropriate analysis/video/data functions
4. Returns an updated pipeline context
5. Exports via `__all__ = ['run']`

This design keeps the code modular while maintaining compatibility with both CLI and notebook workflows.
