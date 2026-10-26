"""Minimal COCO annotation loading for the balloon dataset."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .boxes import coco_to_xyxy
from .config import resolve_dataset_root


SPLITS = ("train", "valid", "test")


def load_coco_split(split, dataset_root=None):
    """Load one split and expose the fields required by the detector."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    root = resolve_dataset_root(dataset_root)
    split_dir = root / split
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    images = {}
    for item in raw.get("images", []):
        image_id = int(item["id"])
        images[image_id] = {
            "id": image_id,
            "file_name": str(item["file_name"]),
            "width": int(item["width"]),
            "height": int(item["height"]),
            "path": split_dir / item["file_name"],
        }

    annotations = []
    annotations_by_image = defaultdict(list)
    for item in raw.get("annotations", []):
        annotation = {
            "id": int(item["id"]),
            "image_id": int(item["image_id"]),
            "category_id": int(item["category_id"]),
            "bbox_coco": [float(value) for value in item["bbox"]],
            "bbox_xyxy": coco_to_xyxy(item["bbox"]),
        }
        annotations.append(annotation)
        annotations_by_image[annotation["image_id"]].append(annotation)

    categories = {
        int(item["id"]): str(item["name"]) for item in raw.get("categories", [])
    }
    balloon_category_ids = {
        category_id
        for category_id, name in categories.items()
        if name.strip().lower() == "balloon"
    }
    if not balloon_category_ids:
        raise ValueError(f"No balloon category found in {annotation_path}")

    return {
        "root": root,
        "split": split,
        "split_dir": split_dir,
        "annotation_path": annotation_path,
        "images": images,
        "annotations": annotations,
        "annotations_by_image": dict(annotations_by_image),
        "categories": categories,
        "balloon_category_ids": balloon_category_ids,
        "raw": raw,
    }


def ground_truth_boxes(coco_data, image_id):
    """Return balloon ground-truth boxes for one image in XYXY format."""
    annotations = coco_data["annotations_by_image"].get(int(image_id), [])
    boxes = [
        item["bbox_xyxy"]
        for item in annotations
        if item["category_id"] in coco_data["balloon_category_ids"]
    ]
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def proposal_cache_path(results_root, split, file_name):
    return Path(results_root) / "proposals" / split / f"{Path(file_name).stem}.pkl"
