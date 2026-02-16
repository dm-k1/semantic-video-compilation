# Semantic Video Compilation via Latent Space Clustering

> **Note**: This repository contains ongoing research shared for master's program admissions review. The work presented here represents current progress on an active research project.

## Purpose

This repository documents an ongoing investigation into semantic organization of video archives using latent space representations. The materials are presented for academic review rather than general distribution.

## The Central Question

How can latent space clustering automate semantic organization of video libraries without supervised labels?

## Overview

The study evaluates a pipeline that combines CLIP-based semantic scoring, percentile aggregation of frame-level evidence, Truncated SVD for variance-preserving reweighting, and adaptive clustering to form balanced compilations. The approach emphasizes interpretability and statistical validation within a notebook-based workflow.

## Current Progress

- Part 1: Video normalization with codec standardization and filename sanitization.
- Part 2: Frame extraction at a fixed sampling rate.
- Part 3: Latent scoring of frames against curated scene statements.
- Part 4: Aggregation of frame scores to video-level representations.
- Part 5: Reweighting and embedding via Truncated SVD.
- Part 6: Strata construction and compilation selection under duration constraints.
- Part 7: Statistical validation against random size-matched samples.

## Project Structure

- `notebooks/Demo.ipynb`: Process outline notebook provided for conceptual review.
- `src/core.py`: Shared utilities for seeding, paths, and reporting.
- `src/data.py`: Data loading and schema validation.
- `src/steps/`: Modular pipeline stages implemented as independent modules.
- `inputs/custom_scenes.csv`: Scene statements used for semantic scoring.
- `inputs/videos/`: Placeholder directory for source footage.
- `outputs/`: Generated artifacts from pipeline execution.
- `scripts/test_installation.py`: Environment verification script.
- `requirements.txt`: Python dependencies.
- `LICENSE`: Rights and usage restrictions.

## Process Outline

A process outline for reviewers is provided in [REPRODUCTION.md](REPRODUCTION.md).

## Status

- Ongoing Research
- Shared for admissions committee review

## Documentation

- [REPRODUCTION.md](REPRODUCTION.md) - Process outline for review.
- [THEORY.md](THEORY.md) - Theoretical framework and methodological rationale.
- [ARCHITECTURE.md](ARCHITECTURE.md) - Pipeline architecture diagram.
- [src/steps/README.md](src/steps/README.md) - Stage-level module descriptions.

## License

All Rights Reserved. See [LICENSE](LICENSE) for terms.

## Acknowledgments

- OpenAI CLIP: https://github.com/openai/CLIP
- FFmpeg: https://ffmpeg.org
- scikit-learn: SVD implementation used in analysis
