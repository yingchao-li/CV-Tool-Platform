"""Unit tests for box geometry, IoU and non-maximum suppression."""

from __future__ import annotations

import numpy as np
import pytest

from cvkit.detection.boxes import (
    coco_to_xyxy,
    iou,
    nms,
    pairwise_iou,
    xyxy_to_coco,
)


def test_iou_identical_boxes_is_one():
    assert iou([0, 0, 10, 10], [[0, 0, 10, 10]])[0] == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou([0, 0, 10, 10], [[20, 20, 30, 30]])[0] == pytest.approx(0.0)


def test_iou_half_overlap():
    # Intersection 50, union 150.
    assert iou([0, 0, 10, 10], [[5, 0, 15, 10]])[0] == pytest.approx(50 / 150)


def test_pairwise_iou_shape():
    boxes_a = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=float)
    boxes_b = np.array([[0, 0, 10, 10]], dtype=float)
    assert pairwise_iou(boxes_a, boxes_b).shape == (2, 1)


def test_nms_keeps_best_and_suppresses_duplicate():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float
    )
    scores = np.array([0.9, 0.8, 0.7])

    kept = nms(boxes, scores, iou_threshold=0.3)
    assert kept.tolist() == [0, 2]


def test_nms_respects_max_detections():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    assert nms(boxes, np.array([0.5, 0.4]), max_detections=1).tolist() == [0]


def test_coco_round_trip():
    xyxy = [10, 20, 40, 60]
    coco = xyxy_to_coco(xyxy)
    assert coco == pytest.approx([10, 20, 30, 40])
    assert coco_to_xyxy(coco) == pytest.approx(xyxy)
