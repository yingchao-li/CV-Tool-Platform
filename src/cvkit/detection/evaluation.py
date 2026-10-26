"""Evaluate proposal quality (MABO) and detector quality (mAP) on a split."""

import argparse
import json
from pathlib import Path

import numpy as np

from .boxes import iou, pairwise_iou, xyxy_to_coco
from .datasets import ground_truth_boxes, load_coco_split
from .detector import detect, load_model_bundle
from .config import DEFAULT_RESULTS_ROOT, ensure_output_tree
from .proposals import load_proposal_cache, read_rgb


IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def compute_mabo(proposals_by_image, ground_truth_by_image):
    best_overlaps = []
    for image_id, gt_boxes in ground_truth_by_image.items():
        proposals = proposals_by_image.get(image_id, np.empty((0, 4), dtype=np.float32))
        if len(gt_boxes) == 0:
            continue
        if len(proposals) == 0:
            best_overlaps.extend([0.0] * len(gt_boxes))
            continue
        overlaps = pairwise_iou(gt_boxes, proposals)
        best_overlaps.extend(overlaps.max(axis=1).tolist())
    return float(np.mean(best_overlaps)) if best_overlaps else 0.0, best_overlaps


def average_precision_at_iou(detections, ground_truth_by_image, threshold):
    total_ground_truth = sum(len(boxes) for boxes in ground_truth_by_image.values())
    if total_ground_truth == 0:
        return 0.0
    matched = {
        image_id: np.zeros(len(boxes), dtype=bool)
        for image_id, boxes in ground_truth_by_image.items()
    }
    ranked = sorted(detections, key=lambda item: item["score"], reverse=True)
    true_positives = np.zeros(len(ranked), dtype=np.float32)
    false_positives = np.zeros(len(ranked), dtype=np.float32)

    for index, detection in enumerate(ranked):
        image_id = detection["image_id"]
        gt_boxes = ground_truth_by_image.get(image_id, np.empty((0, 4), dtype=np.float32))
        if len(gt_boxes) == 0:
            false_positives[index] = 1.0
            continue
        overlaps = iou(detection["bbox_xyxy"], gt_boxes)
        best_index = int(np.argmax(overlaps))
        if overlaps[best_index] >= threshold and not matched[image_id][best_index]:
            true_positives[index] = 1.0
            matched[image_id][best_index] = True
        else:
            false_positives[index] = 1.0

    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recall = cumulative_tp / total_ground_truth
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precision[recall >= point])) if np.any(recall >= point) else 0.0
        for point in recall_points
    ]
    return float(np.mean(interpolated))


def fallback_map(detections, ground_truth_by_image):
    ap_by_iou = {
        f"{threshold:.2f}": average_precision_at_iou(
            detections, ground_truth_by_image, float(threshold)
        )
        for threshold in IOU_THRESHOLDS
    }
    return float(np.mean(list(ap_by_iou.values()))), ap_by_iou


def pycocotools_map(coco_data, detections, category_id):
    """Return official COCO mAP, or None when pycocotools is unavailable."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None
    if not detections:
        return 0.0

    coco_gt = COCO(str(coco_data["annotation_path"]))
    coco_results = [
        {
            "image_id": int(item["image_id"]),
            "category_id": int(category_id),
            "bbox": xyxy_to_coco(item["bbox_xyxy"]),
            "score": float(item["score"]),
        }
        for item in detections
    ]
    coco_dt = coco_gt.loadRes(coco_results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(coco_data["images"])
    evaluator.params.catIds = [int(category_id)]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[0])


def evaluate(args):
    if not 0.0 <= args.nms_threshold <= 1.0:
        raise ValueError("nms_threshold must be in [0, 1]")
    output_dirs = ensure_output_tree()
    split = getattr(args, "split", "test")
    coco = load_coco_split(split, args.dataset_root)
    bundle = load_model_bundle(args.model_path)
    proposal_root = Path(
        getattr(args, "proposal_root", None) or output_dirs["proposals"]
    )

    proposals_by_image = {}
    ground_truth_by_image = {}
    detections = []
    used_category_ids = {
        annotation["category_id"]
        for annotation in coco["annotations"]
        if annotation["category_id"] in coco["balloon_category_ids"]
    }
    category_id = min(used_category_ids or coco["balloon_category_ids"])

    for image_id in sorted(coco["images"]):
        image_record = coco["images"][image_id]
        cache_path = proposal_root / split / f"{Path(image_record['file_name']).stem}.pkl"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing proposal cache {cache_path}. Generate {split} proposals first."
            )
        proposals = load_proposal_cache(cache_path)["boxes_xyxy"]
        gt_boxes = ground_truth_boxes(coco, image_id)
        proposals_by_image[image_id] = proposals
        ground_truth_by_image[image_id] = gt_boxes

        image = read_rgb(image_record["path"])
        boxes, scores = detect(
            image,
            proposals,
            bundle,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            max_detections=args.max_detections,
        )
        for box, score in zip(boxes, scores):
            detections.append(
                {
                    "image_id": int(image_id),
                    "file_name": image_record["file_name"],
                    "category_id": int(category_id),
                    "bbox_xyxy": [float(value) for value in box],
                    "bbox_coco": xyxy_to_coco(box),
                    "score": float(score),
                }
            )
        print(
            f"[evaluate] {image_record['file_name']}: "
            f"proposals={len(proposals)} detections={len(boxes)} gt={len(gt_boxes)}"
        )

    mabo, best_overlaps = compute_mabo(proposals_by_image, ground_truth_by_image)
    approximate_map, ap_by_iou = fallback_map(detections, ground_truth_by_image)
    official_map = None
    if not bool(getattr(args, "force_fallback", False)):
        official_map = pycocotools_map(coco, detections, category_id)
    backend = "pycocotools" if official_map is not None else "fallback_101_point"
    mean_ap = official_map if official_map is not None else approximate_map

    metrics = {
        "bbox_format": "xyxy_exclusive",
        "split": split,
        "mabo": mabo,
        "map_50_95": mean_ap,
        "map_backend": backend,
        "fallback_map_50_95": approximate_map,
        "ap_by_iou_fallback": ap_by_iou,
        "num_images": len(coco["images"]),
        "num_test_images": len(coco["images"]) if split == "test" else None,
        "num_ground_truth_objects": int(sum(len(boxes) for boxes in ground_truth_by_image.values())),
        "num_detections": len(detections),
        "proposal_best_ious": [float(value) for value in best_overlaps],
        "configuration": {
            "split": split,
            "model_path": str(args.model_path),
            "proposal_root": str(proposal_root),
            "score_threshold": args.score_threshold,
            "nms_threshold": args.nms_threshold,
            "max_detections_per_image": args.max_detections,
            "force_fallback": bool(getattr(args, "force_fallback", False)),
        },
    }
    metrics_path = Path(
        getattr(args, "metrics_path", None)
        or (output_dirs["evaluation"] / "metrics.json")
    )
    detections_path = Path(
        getattr(args, "detections_path", None)
        or (output_dirs["evaluation"] / "test_detections.json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    detections_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with detections_path.open("w", encoding="utf-8") as handle:
        json.dump(detections, handle, indent=2)
    print(f"MABO: {mabo:.4f}")
    print(f"mAP@[.50:.95]: {mean_ap:.4f} ({backend})")
    print(f"Saved metrics: {metrics_path}")
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "models" / "balloon_svm.joblib",
    )
    parser.add_argument("--score-threshold", type=float, default=-1000000.0)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--proposal-root", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--detections-path", type=Path, default=None)
    parser.add_argument("--force-fallback", action="store_true")
    return parser


def main(argv=None):
    evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
