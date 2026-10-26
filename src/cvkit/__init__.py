"""CVKit: a modular classical computer vision toolkit.

The package is organised into five independently usable subsystems that
share one data layout, one CLI and one set of conventions:

``cvkit.geometry``
    3D box measurement from depth-camera point clouds with robust plane
    fitting (RANSAC / MLESAC / Preemptive RANSAC).
``cvkit.imaging``
    Computational photography: Bayer demosaicing, white balance, sensor
    linearity, exposure-response estimation and HDR merging.
``cvkit.retrieval``
    Instance retrieval with RootSIFT, VLAD, power/intra normalisation and
    exemplar SVMs, plus a reproducible ablation suite.
``cvkit.detection``
    Selective Search proposals, HOG features, a linear SVM detector and
    COCO-style evaluation with a validation-driven parameter search.
``cvkit.faces``
    Face detection, tracking, alignment, embedding-based recognition,
    clustering and open-set rejection (SPL / MPL).
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
