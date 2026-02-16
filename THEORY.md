# Theoretical Background

## Theoretical Framework

### 1. Introduction

CLIP (Contrastive Language-Image Pre-training) constructs a joint embedding space for images and text through contrastive learning on large-scale paired data. The image encoder and text encoder each output a 512-dimensional embedding for the ViT-B/16 variant. Semantic similarity is operationalized with cosine similarity, defined as the normalized dot product between the image and text embeddings.

### 2. Methodology

#### 2.1 Frame-Level Scoring

Frames are extracted from each video and scored against each scene statement, producing a matrix in which rows correspond to frames and columns correspond to scene statements. Each entry represents a cosine similarity score in the range $[0, 1]$.

#### 2.2 Aggregation

Aggregation transforms frame-level scores into video-level representations. A percentile operator is used to retain high-salience evidence while reducing sensitivity to low-salience frames. The 95th percentile is used to retain the top 5% of frame evidence for each scene statement.

#### 2.3 Adaptive Clustering

Videos are embedded in a lower-dimensional space and partitioned into $K$ strata by minimizing the maximum distance from any video to its nearest prototype. A greedy farthest-first traversal is used to approximate the $K$-center objective, yielding a $2$-approximation bound on the optimal maximum distance. The value of $K$ is selected by sweeping candidate values and stopping when the maximum coverage distance falls below a fixed threshold.

#### 2.4 Statistical Validation

The interchangeability hypothesis is formalized as a null-hypothesis test. The null hypothesis posits that each compilation is a random size-matched sample from the global pool. The alternative hypothesis asserts systematic bias in the compilation selection.

Multiple test statistics are evaluated, including variance capture, mean pairwise distance, centroid shift, and maximum mean discrepancy. A permutation procedure generates null distributions for each statistic and combines deviations via the maximum standardized score to control for multiple comparisons.

### 3. Dimensionality Reduction

#### 3.1 Motivation

The video-by-scene score matrix contains redundancy, sparse signals, and high dimensionality. Dimensionality reduction is used to preserve dominant structure while attenuating noise.

#### 3.2 Truncated SVD

Truncated SVD decomposes the score matrix into orthogonal components ordered by explained variance. The retained components capture dominant semantic directions and reduce redundancy among correlated scene statements.

#### 3.3 Variance Preservation

The number of retained components is selected to preserve a target fraction of variance. Components below this threshold are treated as noise and excluded from subsequent analysis.

#### 3.4 Communality Reweighting

Scene statements are reweighted by their communalities, defined as the sum of squared loadings across retained components. This reweighting reduces the influence of statements with weak alignment to the dominant latent structure while preserving contributions from informative statements.

**Last Updated:** February 15, 2026  
**Author:** Diemithry Kloppenburg
