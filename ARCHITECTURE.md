# Pipeline Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC VIDEO COMPILATION PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT STAGE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Raw Videos                             Scene Statements
    ┌─────────────┐                       ┌────────────────────────┐
    │ video1.mp4  │                       │ "Small boat on         │
    │ video2.mov  │                       │  open water"           │
    │ video3.avi  │                       │                        │
    │    ...      │                       │ "Organised tent rows"  │
    │ videoN.mkv  │                       │                        │
    └─────────────┘                       │ "Border crossing gate" │
           │                              │     ...                │
           ↓                              └────────────────────────┘
                                                   ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                        STEP 1: VIDEO PREPROCESSING                          │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────────────────┐
    │  Step 1a: Normalise Videos (FFmpeg)                       │
    │  ─────────────────────────────────────                    │
    │  • Sanitise filenames (ASCII-safe)                        │
    │  • Re-encode to h264/aac                                  │
    │  • Optionally resize to target resolution                 │
    │                                                           │
    │  Output: normalized_videos/                               │
    └───────────────────────────────────────────────────────────┘
                              ↓
    ┌───────────────────────────────────────────────────────────┐
    │  Step 1b: Extract Frames                                  │
    │  ─────────────────────────                                │
    │  • Sample at FPS=3 (configurable)                         │
    │  • Save as JPEG images                                    │
    │  • Create frames_df DataFrame                             │
    │                                                           │
    │  Output: frames/video_id/frame_*.jpg                      │
    └───────────────────────────────────────────────────────────┘
                              ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                      STEP 2: SEMANTIC ENCODING                              │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────────────────┐
    │  CLIP Encoding (ViT-B/16)                                 │
    │  ─────────────────────────                                │
    │  • Encode frames → 512-dim embeddings                     │
    │  • Encode text prompts → 512-dim embeddings               │
    │  • Compute cosine similarity scores                       │
    │                                                           │
    │  Matrix: frame_scores[n_frames × n_scenes]                │
    │          Each entry = similarity ∈ [0, 1]                 │
    └───────────────────────────────────────────────────────────┘
                              ↓

         Frame-Level Scores (example for 1 video)
         ┌───────────────────────────────────────┐
         │       Scene 1  Scene 2  Scene 3  ...  │
         │ Frame 1:  0.23    0.45    0.12        │
         │ Frame 2:  0.67    0.34    0.89        │
         │ Frame 3:  0.12    0.78    0.45        │
         │   ...                                 │
         └───────────────────────────────────────┘
                              ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                      STEP 3: AGGREGATION                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────────────────┐
    │  Percentile Pooling (95th percentile)                     │
    │  ──────────────────────────────────────                   │
    │  • For each video, for each scene:                        │
    │    v_j = Q_0.95({scores for all frames})                  │
    │  • Captures peak moments, robust to noise                 │
    │                                                           │
    │  Matrix: video_scores[n_videos × n_scenes]                │
    └───────────────────────────────────────────────────────────┘
                              ↓

         Video-Level Scores
         ┌───────────────────────────────────────┐
         │         Scene 1  Scene 2  Scene 3 ... │
         │ Video 1:  0.65    0.78    0.45        │
         │ Video 2:  0.23    0.89    0.67        │
         │ Video 3:  0.91    0.34    0.23        │
         │   ...                                 │
         └───────────────────────────────────────┘
                              ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                  STEP 4: DIMENSIONALITY REDUCTION                           │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────────────────┐
    │  Truncated SVD                                            │
    │  ──────────────                                           │
    │  X = U Σ V^T                                              │
    │                                                           │
    │  1. Decompose the video_scores matrix                     │
    │  2. Retain top k components (95% variance)                │
    │  3. Compute communalities: h²_j = Σ(v_ji²)                │
    │  4. Reweight scenes: w_j = h²_j^β (β=0.5)                 │
    │                                                           │
    │  Output: svd_scores[n_videos × k]                         │
    │          k << n_scenes (e.g., 16 vs 50)                   │
    └───────────────────────────────────────────────────────────┘
                              ↓

         Reduced Embedding (k=16 dimensions)
         ┌──────────────────────────────────────────┐
         │        PC1   PC2   PC3  ...  PC16       │
         │ Video 1: 0.23  0.45 -0.12 ... 0.34      │
         │ Video 2: 0.67 -0.34  0.89 ... -0.12     │
         │ Video 3: -0.12 0.78  0.45 ... 0.67      │
         │   ...                                   │
         └──────────────────────────────────────────┘
                              ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 5: ADAPTIVE CLUSTERING                              │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────────────────┐
    │  Greedy K-Centre (Farthest-First Traversal)               │
    │  ────────────────────────────────────────                 │
    │  1. Auto-select K via coverage sweep                      │
    │     - Try K = 2, 3, 4, ...                                │
    │     - Stop when max_distance ≤ threshold                  │
    │                                                           │
    │  2. Build strata:                                         │
    │     a. Select the medoid as the prototype                 │
    │     b. Add the farthest point from existing prototypes    │
    │     c. Repeat until K prototypes chosen                   │
    │                                                           │
    │  3. Assign videos to the nearest prototype                │
    │                                                           │
    │  Output: stratum_labels[n_videos]                         │
    │          prototype_indices[K]                             │
    └───────────────────────────────────────────────────────────┘
                              ↓

         Strata Assignment (example K=12)
         ┌────────────────────────────────────────┐
         │ Stratum 0: [video_1, video_7, ...]     │
         │ Stratum 1: [video_3, video_12, ...]    │
         │ Stratum 2: [video_5, video_9, ...]     │
         │     ...                                │
         │ Stratum 11: [video_4, video_15, ...]   │
         └────────────────────────────────────────┘
                              ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 6: COMPILATION SELECTION                            │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────────────────┐
    │  Multi-Objective Optimisation                             │
    │  ─────────────────────────────                            │
    │  Constraints:                                             │
    │  • Duration: 55-65 seconds per compilation                │
    │  • Diversity: Cover all strata                            │
    │  • Balance: Minimise overuse of any video                 │
    │                                                           │
    │  Algorithm: Greedy with usage penalties                   │
    │  • Score each candidate by:                               │
    │    - Stratum priority (under-represented strata first)    │
    │    - Usage penalty (penalise frequently selected videos)  │
    │  • Select until duration satisfied                        │
    │                                                           │
    │  Output: compilations[n_compilations]                     │
    │          Each = list of {video_id, stratum_id}            │
    └───────────────────────────────────────────────────────────┘
                              ↓

         Generated Compilations
         ┌────────────────────────────────────────┐
         │ Compilation 1:                         │
         │   [video_3, video_7, video_12, ...]    │
         │   Duration: 58.3s                      │
         │   Strata coverage: 12/12               │
         │                                        │
         │ Compilation 2:                         │
         │   [video_1, video_9, video_14, ...]    │
         │   Duration: 61.7s                      │
         │   Strata coverage: 12/12               │
         │     ...                                │
         └────────────────────────────────────────┘
                              ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                      USE CASE VALIDATION                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────┐
    │  Built-in Validation Checks                                │
    │  ─────────────────────────────                             │
    │  The pipeline includes checks to verify that               │
    │  compilations represent the target dataset:                │
    │                                                            │
    │  1. Coverage: Do all semantic groups appear?               │
    │  2. Balance: Are videos distributed evenly?                │
    │  3. Distinctness: Do compilations differ from random?      │
    │                                                            │
    │  These checks support the assessment of whether the        │
    │  approach fits the target dataset.                         │
    └────────────────────────────────────────────────────────────┘
                              ↓


┌─────────────────────────────────────────────────────────────────────────────┐
│                              FINAL OUTPUT                                   │
└─────────────────────────────────────────────────────────────────────────────┘

    Deliverables
    ┌───────────────────────────────────────────────────────────┐
    │  • N balanced video compilations (e.g., 10)               │
    │  • Each is 55-65 seconds long                             │
    │  • Each covering all strata (diversity guaranteed)        │
    │  • Statistically validated as representative samples      │
    │                                                           │
    │  Use cases:                                               │
    │  • Research stimulus sets for perception studies          │
    │  • Balanced training data for ML models                   │
    │  • Automated content curation                             │
    └───────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════

KEY INNOVATIONS:

- Zero-shot semantic search (no training data required)
- Percentile aggregation (robust to noise, captures peaks)
- SVD with communality reweighting (signal vs noise separation)
- Adaptive K-selection (data-driven cluster count)
- Greedy K-Centre (2-approximation, diversity focus)
- Built-in validation checks (coverage, balance, distinctness)

═══════════════════════════════════════════════════════════════════════════════

COMPUTATIONAL COMPLEXITY:

Step 1: O(N) FFmpeg operations (parallelizable)
Step 2: O(F·d) CLIP encoding, F = frames, d = embed dim
Step 3: O(F·M) aggregation, M = scenes
Step 4: O(N·M·k) SVD, k = components
Step 5: O(N²·K) clustering (distance computations)
Step 6: O(N_C·N·K) selection, N_C = compilations

Total: Dominated by Step 2 (CLIP encoding)
       GPU acceleration is highly recommended

═══════════════════════════════════════════════════════════════════════════════
```
