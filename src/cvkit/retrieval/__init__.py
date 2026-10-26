"""Image retrieval with RootSIFT and VLAD.

The pipeline follows the classic instance-retrieval recipe:

1. detect SIFT keypoints and describe them with RootSIFT (Hellinger
   normalised SIFT);
2. build a visual codebook with mini-batch k-means;
3. encode every image as a VLAD vector (residuals around each visual word);
4. normalise (power / intra-normalisation) and optionally re-project with
   PCA whitening;
5. rank with cosine distance, optionally after an exemplar SVM or average
   query expansion.

``experiments`` contains the ablation drivers that produced the numbers in
``docs/benchmarks.md``.
"""

from __future__ import annotations

from .pipeline import (
    assignments,
    computeDescs,
    dictionary,
    distances,
    esvm,
    evaluate,
    getFiles,
    loadRandomDescriptors,
    nearest_cluster_indices,
    normalize_vlad,
    raw_vlad,
    toBinary,
    vlad,
    vlad_both,
)

__all__ = [
    "assignments",
    "computeDescs",
    "dictionary",
    "distances",
    "esvm",
    "evaluate",
    "getFiles",
    "loadRandomDescriptors",
    "nearest_cluster_indices",
    "normalize_vlad",
    "raw_vlad",
    "toBinary",
    "vlad",
    "vlad_both",
]
