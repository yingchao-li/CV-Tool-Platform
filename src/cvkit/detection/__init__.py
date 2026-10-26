"""Object detection with Selective Search proposals and a HOG/SVM classifier.

This is a CPU-friendly classical detection pipeline:

``selective_search``
    Felzenszwalb segmentation followed by similarity-based region merging.
``proposals``
    Proposal generation and a parameter-aware cache.
``features`` / ``trainer``
    Fixed-size HOG descriptors and a linear SVM trained on IoU-labelled
    proposals.
``detector``
    Score thresholding, non-maximum suppression and visualisation.
``evaluation``
    Proposal MABO and mAP@[.50:.95] with a dependency-free fallback.
``tuning``
    Two-stage, validation-driven parameter search.
"""

from __future__ import annotations

from .boxes import coco_to_xyxy, iou, nms, pairwise_iou, xyxy_to_coco
from .datasets import ground_truth_boxes, load_coco_split
from .detector import detect, load_model_bundle
from .evaluation import compute_mabo, evaluate
from .features import FeatureConfig, extract_features
from .proposals import generate_proposals_for_image, load_proposal_cache
from .selective_search import selective_search
from .trainer import train

__all__ = [
    "FeatureConfig",
    "coco_to_xyxy",
    "compute_mabo",
    "detect",
    "evaluate",
    "extract_features",
    "generate_proposals_for_image",
    "ground_truth_boxes",
    "iou",
    "load_coco_split",
    "load_model_bundle",
    "load_proposal_cache",
    "nms",
    "pairwise_iou",
    "selective_search",
    "train",
    "xyxy_to_coco",
]
