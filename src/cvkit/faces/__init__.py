"""Face detection, embedding, recognition and open-set analysis.

Components
----------
``detector``
    MTCNN detection with template-matching tracking and landmark-based
    alignment.
``recognition``
    FaceNet embeddings, a k-NN gallery recogniser and a k-means clustering
    model for label-free re-identification.
``dir_eval``
    Detection-and-identification rate (DIR) evaluation over similarity
    thresholds.
``openset``
    SPL / MPL open-set training and rejection scoring.
``train`` / ``infer``
    Command-line drivers for building galleries and running inference.
"""

from __future__ import annotations

from .config import Config, ReIdMode, enum_choices
from .detector import FaceDetectionResult, FaceDetector
from .dir_eval import OpenSetEvaluation
from .neighbors import NearestNeighborClassifier
from .openset import mpl_training, spl_training
from .recognition import FaceClustering, FaceNet, FaceRecognizer

__all__ = [
    "Config",
    "FaceClustering",
    "FaceDetectionResult",
    "FaceDetector",
    "FaceNet",
    "FaceRecognizer",
    "NearestNeighborClassifier",
    "OpenSetEvaluation",
    "ReIdMode",
    "enum_choices",
    "mpl_training",
    "spl_training",
]
