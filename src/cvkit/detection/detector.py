"""Run the trained balloon detector on one image."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .boxes import nms, xyxy_to_coco
from .features import FeatureConfig, extract_features
from .config import DEFAULT_RESULTS_ROOT, ensure_output_tree
from .proposals import (
    generate_proposals_for_image,
    load_proposal_cache,
    read_rgb,
)


def load_model_bundle(model_path):
    bundle = joblib.load(model_path)
    required = {"classifier", "feature_config", "bbox_format"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing: {sorted(missing)}")
    if bundle["bbox_format"] != "xyxy_exclusive":
        raise ValueError("Model uses an unsupported bbox format")
    return bundle


def classifier_scores(classifier, features):
    """Return a higher-is-more-balloon score for each feature vector."""
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)
        classes = np.asarray(classifier.classes_)
        positive_column = int(np.flatnonzero(classes == 1)[0])
        return probabilities[:, positive_column].astype(np.float32)
    if hasattr(classifier, "decision_function"):
        return np.asarray(classifier.decision_function(features), dtype=np.float32).reshape(-1)
    return np.asarray(classifier.predict(features), dtype=np.float32).reshape(-1)


def detect(image, proposals, bundle, *, score_threshold=0.0, nms_threshold=0.3, max_detections=100):
    config = FeatureConfig.from_dict(bundle["feature_config"])
    features, boxes, _ = extract_features(image, proposals, config)
    if len(features) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    scores = classifier_scores(bundle["classifier"], features)
    selected = scores >= score_threshold
    boxes = boxes[selected]
    scores = scores[selected]
    if len(boxes) == 0:
        return boxes, scores

    kept = nms(boxes, scores, nms_threshold, max_detections=max_detections)
    return boxes[kept], scores[kept]


def find_cached_proposals(image_path, proposals_root=None):
    proposals_root = Path(proposals_root or (DEFAULT_RESULTS_ROOT / "proposals"))
    candidates = sorted(proposals_root.glob(f"*/{image_path.stem}.pkl"))
    for candidate in candidates:
        payload = load_proposal_cache(candidate)
        if payload.get("file_name") == image_path.name:
            return payload, candidate
    return None, None


def save_visualization(image, boxes, scores, output_path):
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        rectangle = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rectangle)
        ax.text(
            x1,
            max(0, y1 - 3),
            f"balloon {score:.3f}",
            color="white",
            fontsize=8,
            bbox={"facecolor": "red", "alpha": 0.75, "pad": 2},
        )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def run_inference(args):
    image_path = args.input.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    output_dirs = ensure_output_tree()
    image = read_rgb(image_path)
    bundle = load_model_bundle(args.model_path)

    cache_path = None
    cache = None
    if args.reuse_proposals:
        cache, cache_path = find_cached_proposals(image_path, output_dirs["proposals"])
    if cache is not None:
        proposals = cache["boxes_xyxy"]
        print(f"Loaded {len(proposals)} cached proposals from {cache_path}")
    else:
        proposals = generate_proposals_for_image(
            image,
            scale=args.scale,
            sigma=args.sigma,
            min_size=args.min_size,
            max_proposals=args.max_proposals,
        )
        print(f"Generated {len(proposals)} proposals")

    boxes, scores = detect(
        image,
        proposals,
        bundle,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
    )
    image_output = output_dirs["detections"] / f"{image_path.stem}_detection.jpg"
    json_output = output_dirs["detections"] / f"{image_path.stem}_detection.json"
    save_visualization(image, boxes, scores, image_output)
    payload = {
        "input_image": str(image_path),
        "bbox_format": "xyxy_exclusive",
        "score_type": "probability" if hasattr(bundle["classifier"], "predict_proba") else "decision_function",
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "detections": [
            {
                "label": "balloon",
                "score": float(score),
                "bbox_xyxy": [float(value) for value in box],
                "bbox_coco": xyxy_to_coco(box),
            }
            for box, score in zip(boxes, scores)
        ],
    }
    with json_output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved {len(boxes)} detections to {image_output}")
    print(f"Saved detection data to {json_output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "models" / "balloon_svm.joblib",
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--reuse-proposals", action="store_true")
    parser.add_argument("--scale", type=float, default=500.0)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--max-proposals", type=int, default=2000)
    parser.add_argument("--max-detections", type=int, default=100)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.nms_threshold <= 1.0:
        raise ValueError("nms_threshold must be in [0, 1]")
    run_inference(args)


if __name__ == "__main__":
    main()
