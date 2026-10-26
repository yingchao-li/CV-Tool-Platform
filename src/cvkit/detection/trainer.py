"""Create IoU-labelled samples, extract HOG features, and train an SVM."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .boxes import clip_box, is_valid_box, pairwise_iou
from .datasets import ground_truth_boxes, load_coco_split
from .features import FeatureConfig, extract_features
from .config import ensure_output_tree
from .proposals import load_proposal_cache, read_rgb


def select_training_samples(
    proposals,
    ground_truth,
    *,
    tp=0.75,
    tn=0.25,
    max_positive=32,
    max_negative=96,
):
    """Return selected boxes, labels, and maximum IoU values."""
    proposals = np.asarray(proposals, dtype=np.float32).reshape(-1, 4)
    ground_truth = np.asarray(ground_truth, dtype=np.float32).reshape(-1, 4)
    if len(proposals) == 0:
        return proposals, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    if len(ground_truth):
        overlaps = pairwise_iou(proposals, ground_truth)
        max_ious = overlaps.max(axis=1)
    else:
        max_ious = np.zeros(len(proposals), dtype=np.float32)

    positive_indices = np.flatnonzero(max_ious >= tp)
    negative_indices = np.flatnonzero(max_ious <= tn)

    # Keep the strongest positives and the hardest allowed negatives.
    positive_indices = positive_indices[np.argsort(max_ious[positive_indices])[::-1]]
    negative_indices = negative_indices[np.argsort(max_ious[negative_indices])[::-1]]
    if max_positive >= 0:
        positive_indices = positive_indices[:max_positive]
    if max_negative >= 0:
        negative_indices = negative_indices[:max_negative]

    selected_indices = np.concatenate([positive_indices, negative_indices])
    labels = np.concatenate(
        [
            np.ones(len(positive_indices), dtype=np.int64),
            np.zeros(len(negative_indices), dtype=np.int64),
        ]
    )
    return proposals[selected_indices], labels, max_ious[selected_indices]


def jitter_ground_truth_boxes(
    ground_truth,
    image_width,
    image_height,
    *,
    rng,
    count_per_box=2,
    fraction=0.08,
):
    """Create small GT perturbations for training augmentation only."""
    ground_truth = np.asarray(ground_truth, dtype=np.float32).reshape(-1, 4)
    jittered = []
    for box in ground_truth:
        width = max(float(box[2] - box[0]), 1.0)
        height = max(float(box[3] - box[1]), 1.0)
        for _ in range(max(0, int(count_per_box))):
            offsets = rng.normal(0.0, fraction, size=4) * np.asarray(
                [width, height, width, height], dtype=np.float32
            )
            candidate = clip_box(
                box + offsets,
                image_width,
                image_height,
            )
            if is_valid_box(candidate):
                jittered.append(candidate)
    if not jittered:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(jittered, dtype=np.float32).reshape(-1, 4)


def train(args):
    if not 0.0 <= args.tn < args.tp <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= tn < tp <= 1")
    if args.max_positive == 0 or args.max_positive < -1:
        raise ValueError("max_positive must be positive, or -1 for no limit")
    if args.max_negative == 0 or args.max_negative < -1:
        raise ValueError("max_negative must be positive, or -1 for no limit")

    output_dirs = ensure_output_tree()
    coco = load_coco_split("train", args.dataset_root)
    proposal_root = Path(
        getattr(args, "proposal_root", None) or output_dirs["proposals"]
    )
    config = FeatureConfig(
        resize_height=args.resize_size[0],
        resize_width=args.resize_size[1],
        orientations=args.orientations,
        pixels_per_cell=args.pixels_per_cell,
        cells_per_block=args.cells_per_block,
        min_crop_size=args.min_crop_size,
    )

    feature_batches = []
    label_batches = []
    image_summaries = []
    max_positive = -1 if args.max_positive == -1 else args.max_positive
    max_negative = -1 if args.max_negative == -1 else args.max_negative

    for image_id in sorted(coco["images"]):
        image_record = coco["images"][image_id]
        cache_path = proposal_root / "train" / f"{Path(image_record['file_name']).stem}.pkl"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing proposal cache {cache_path}. Generate train proposals first."
            )
        proposals = load_proposal_cache(cache_path)["boxes_xyxy"]
        gt_boxes = ground_truth_boxes(coco, image_id)
        sample_boxes, labels, sample_ious = select_training_samples(
            proposals,
            gt_boxes,
            tp=args.tp,
            tn=args.tn,
            max_positive=max_positive,
            max_negative=max_negative,
        )
        proposal_sample_count = len(labels)

        # Ground-truth and jittered boxes are training augmentation. They are
        # intentionally not written into or reported as Selective Search proposals.
        augmented_boxes = []
        if bool(getattr(args, "add_gt_as_positive", False)):
            augmented_boxes.extend(gt_boxes)
        if bool(getattr(args, "jitter_gt_boxes", False)):
            rng = np.random.default_rng(int(args.seed) + int(image_id))
            augmented_boxes.extend(
                jitter_ground_truth_boxes(
                    gt_boxes,
                    image_record["width"],
                    image_record["height"],
                    rng=rng,
                    count_per_box=int(getattr(args, "jitter_count_per_box", 2)),
                    fraction=float(getattr(args, "jitter_fraction", 0.08)),
                )
            )
        if augmented_boxes:
            augmented_boxes = np.asarray(augmented_boxes, dtype=np.float32).reshape(-1, 4)
            sample_boxes = np.vstack([sample_boxes, augmented_boxes])
            labels = np.concatenate(
                [labels, np.ones(len(augmented_boxes), dtype=np.int64)]
            )
            sample_ious = np.concatenate(
                [sample_ious, np.ones(len(augmented_boxes), dtype=np.float32)]
            )

        image = read_rgb(image_record["path"])
        features, _, valid_indices = extract_features(image, sample_boxes, config)
        valid_labels = labels[valid_indices]
        if len(features):
            feature_batches.append(features)
            label_batches.append(valid_labels)
        summary = {
            "image_id": image_id,
            "file_name": image_record["file_name"],
            "positive": int(np.sum(valid_labels == 1)),
            "negative": int(np.sum(valid_labels == 0)),
            "ignored": int(len(proposals) - proposal_sample_count),
            "augmented_positive": int(len(labels) - proposal_sample_count),
            "selected_mean_iou": float(np.mean(sample_ious[valid_indices])) if len(valid_indices) else 0.0,
        }
        image_summaries.append(summary)
        print(
            f"[samples] {summary['file_name']}: "
            f"positive={summary['positive']} negative={summary['negative']}"
        )

    if not feature_batches:
        raise RuntimeError("No valid training samples were extracted")
    features = np.vstack(feature_batches)
    labels = np.concatenate(label_batches)
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    if args.negative_positive_ratio > 0 and len(positive_indices):
        maximum_negatives = int(np.ceil(args.negative_positive_ratio * len(positive_indices)))
        if len(negative_indices) > maximum_negatives:
            rng = np.random.default_rng(args.seed)
            negative_indices = np.sort(
                rng.choice(negative_indices, size=maximum_negatives, replace=False)
            )
            balanced_indices = np.sort(np.concatenate([positive_indices, negative_indices]))
            features = features[balanced_indices]
            labels = labels[balanced_indices]
    classes, counts = np.unique(labels, return_counts=True)
    if set(classes.tolist()) != {0, 1}:
        raise RuntimeError(
            f"Training requires both background and balloon samples; got {dict(zip(classes, counts))}"
        )

    class_weight = getattr(args, "class_weight", "balanced")
    if isinstance(class_weight, str) and class_weight.lower() == "none":
        class_weight = None
    classifier = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=args.c,
            class_weight=class_weight,
            max_iter=args.max_iter,
            random_state=args.seed,
        ),
    )
    classifier.fit(features, labels)

    model_path = Path(
        getattr(args, "model_path", None)
        or (output_dirs["models"] / "balloon_svm.joblib")
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": 1,
        "classifier": classifier,
        "feature_config": config.to_dict(),
        "bbox_format": "xyxy_exclusive",
        "classes": {"background": 0, "balloon": 1},
        "training": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tp": args.tp,
            "tn": args.tn,
            "max_positive_per_image": args.max_positive,
            "max_negative_per_image": args.max_negative,
            "negative_positive_ratio": args.negative_positive_ratio,
            "class_weight": class_weight,
            "svm_C": args.c,
            "add_gt_as_positive": bool(getattr(args, "add_gt_as_positive", False)),
            "jitter_gt_boxes": bool(getattr(args, "jitter_gt_boxes", False)),
            "num_samples": int(len(labels)),
            "num_positive": int(np.sum(labels == 1)),
            "num_negative": int(np.sum(labels == 0)),
            "feature_length": int(features.shape[1]),
            "images": image_summaries,
        },
    }
    joblib.dump(bundle, model_path)
    print(f"Saved model: {model_path}")
    print(
        f"Training samples: {len(labels)} "
        f"(positive={bundle['training']['num_positive']}, negative={bundle['training']['num_negative']})"
    )
    return model_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--tp", type=float, default=0.75)
    parser.add_argument("--tn", type=float, default=0.25)
    parser.add_argument("--max-positive", type=int, default=32)
    parser.add_argument("--max-negative", type=int, default=96)
    parser.add_argument(
        "--negative-positive-ratio",
        type=float,
        default=5.0,
        help="Global maximum negative/positive ratio; <=0 disables this cap",
    )
    parser.add_argument("--resize-size", nargs=2, type=int, default=(64, 64), metavar=("H", "W"))
    parser.add_argument("--orientations", type=int, default=9)
    parser.add_argument("--pixels-per-cell", type=int, default=8)
    parser.add_argument("--cells-per-block", type=int, default=2)
    parser.add_argument("--min-crop-size", type=int, default=4)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--add-gt-as-positive", action="store_true")
    parser.add_argument("--jitter-gt-boxes", action="store_true")
    parser.add_argument("--jitter-count-per-box", type=int, default=2)
    parser.add_argument("--jitter-fraction", type=float, default=0.08)
    parser.add_argument("--proposal-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
