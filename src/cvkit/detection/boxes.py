"""Bounding-box conversion, overlap, and suppression utilities.

All internal boxes use ``[x1, y1, x2, y2]`` with an exclusive bottom-right
corner. COCO boxes are converted at file boundaries from/to ``[x, y, w, h]``.
"""

import numpy as np


def coco_to_xyxy(box):
    x, y, width, height = (float(value) for value in box)
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def xyxy_to_coco(box):
    x1, y1, x2, y2 = (float(value) for value in box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def clip_box(box, width, height):
    x1, y1, x2, y2 = (float(value) for value in box)
    return np.asarray(
        [
            np.clip(x1, 0, width),
            np.clip(y1, 0, height),
            np.clip(x2, 0, width),
            np.clip(y2, 0, height),
        ],
        dtype=np.float32,
    )


def is_valid_box(box, min_size=1.0):
    x1, y1, x2, y2 = (float(value) for value in box)
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def box_area(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    boxes = np.atleast_2d(boxes)
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return widths * heights


def iou(box, boxes):
    """Compute IoU between one XYXY box and one or more XYXY boxes."""
    box = np.asarray(box, dtype=np.float32).reshape(4)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    boxes = np.atleast_2d(boxes)

    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection_size = np.maximum(0.0, bottom_right - top_left)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    union = box_area(box.reshape(1, 4))[0] + box_area(boxes) - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def pairwise_iou(boxes_a, boxes_b):
    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    return np.vstack([iou(box, boxes_b) for box in boxes_a])


def nms(boxes, scores, iou_threshold=0.3, max_detections=None):
    """Greedy non-maximum suppression; return indices into the inputs."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")

    order = np.argsort(scores)[::-1]
    kept = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if max_detections is not None and len(kept) >= max_detections:
            break
        if order.size == 1:
            break
        remaining = order[1:]
        overlaps = iou(boxes[current], boxes[remaining])
        order = remaining[overlaps <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)
