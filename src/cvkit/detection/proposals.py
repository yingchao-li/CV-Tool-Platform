"""Generate and cache Selective Search proposals for balloon images."""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import skimage.color
import skimage.io
import skimage.util

from .selective_search import selective_search

from .boxes import clip_box, is_valid_box
from .datasets import SPLITS, load_coco_split, proposal_cache_path
from .config import ensure_output_tree


def read_rgb(image_path):
    image = skimage.io.imread(image_path)
    if image.ndim == 2:
        image = skimage.color.gray2rgb(image)
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape {image.shape}: {image_path}")
    if image.shape[2] == 4:
        image = skimage.color.rgba2rgb(image)
    if image.shape[2] != 3:
        raise ValueError(f"Expected three color channels: {image_path}")
    if image.dtype != np.uint8:
        image = skimage.util.img_as_ubyte(image)
    return image


def regions_to_boxes(regions, image_width, image_height, max_proposals=None):
    """Convert Selective Search rectangles to unique, clipped XYXY boxes."""
    boxes = []
    seen = set()
    for region in regions:
        x, y, width, height = (float(value) for value in region["rect"])
        # Selective Search defines width = max_x - min_x from inclusive pixel extrema.
        # Add one here to obtain the exclusive XYXY convention used in this pipeline.
        box = clip_box(
            [x, y, x + width + 1.0, y + height + 1.0],
            image_width,
            image_height,
        )
        if not is_valid_box(box):
            continue
        key = tuple(float(value) for value in box)
        if key in seen:
            continue
        seen.add(key)
        boxes.append(box)
        if max_proposals is not None and max_proposals > 0 and len(boxes) >= max_proposals:
            break
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def generate_proposals_for_image(
    image,
    scale=500.0,
    sigma=0.8,
    min_size=20,
    max_proposals=2000,
):
    # The segmenter concatenates integer labels as a fourth channel. Supplying
    # normalized floats prevents that concatenation from promoting uint8 RGB
    # to int64, which would distort the later rgb2hsv conversion.
    search_image = skimage.util.img_as_float32(image)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Applying `local_binary_pattern` to floating-point images",
            category=UserWarning,
        )
        _, regions = selective_search(
            search_image,
            scale=scale,
            sigma=sigma,
            min_size=min_size,
        )
    return regions_to_boxes(
        regions,
        image_width=image.shape[1],
        image_height=image.shape[0],
        max_proposals=max_proposals,
    )


def save_proposal_cache(path, *, boxes, image_record, split, parameters):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "bbox_format": "xyxy_exclusive",
        "split": split,
        "image_id": int(image_record["id"]),
        "file_name": image_record["file_name"],
        "image_width": int(image_record["width"]),
        "image_height": int(image_record["height"]),
        "parameters": dict(parameters),
        "boxes_xyxy": np.asarray(boxes, dtype=np.float32),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_proposal_cache(path):
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("bbox_format") != "xyxy_exclusive":
        raise ValueError(f"Unsupported proposal bbox format in {path}")
    payload["boxes_xyxy"] = np.asarray(payload["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
    return payload


def generate_split(
    split,
    *,
    dataset_root=None,
    results_root=None,
    scale=500.0,
    sigma=0.8,
    min_size=20,
    max_proposals=2000,
    overwrite=False,
):
    coco = load_coco_split(split, dataset_root)
    output_dirs = ensure_output_tree(results_root)
    parameters = {
        "scale": float(scale),
        "sigma": float(sigma),
        "min_size": int(min_size),
        "max_proposals": -1 if max_proposals is None else int(max_proposals),
    }
    summary = {"generated": 0, "skipped": 0, "proposals": 0}

    for image_id in sorted(coco["images"]):
        image_record = coco["images"][image_id]
        cache_path = proposal_cache_path(output_dirs["root"], split, image_record["file_name"])
        if cache_path.exists() and not overwrite:
            cached = load_proposal_cache(cache_path)
            summary["skipped"] += 1
            summary["proposals"] += len(cached["boxes_xyxy"])
            print(f"[skip] {split}/{image_record['file_name']}: {len(cached['boxes_xyxy'])} proposals")
            continue

        image = read_rgb(image_record["path"])
        boxes = generate_proposals_for_image(
            image,
            scale=parameters["scale"],
            sigma=parameters["sigma"],
            min_size=parameters["min_size"],
            max_proposals=max_proposals,
        )
        save_proposal_cache(
            cache_path,
            boxes=boxes,
            image_record=image_record,
            split=split,
            parameters=parameters,
        )
        summary["generated"] += 1
        summary["proposals"] += len(boxes)
        print(f"[done] {split}/{image_record['file_name']}: {len(boxes)} proposals")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=500.0)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--max-proposals", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_proposals == 0 or args.max_proposals < -1:
        raise ValueError("max_proposals must be positive, or -1 for no limit")
    max_proposals = None if args.max_proposals == -1 else args.max_proposals
    splits = SPLITS if args.split == "all" else (args.split,)
    for split in splits:
        summary = generate_split(
            split,
            dataset_root=args.dataset_root,
            scale=args.scale,
            sigma=args.sigma,
            min_size=args.min_size,
            max_proposals=max_proposals,
            overwrite=args.overwrite,
        )
        print(f"{split}: {summary}")


if __name__ == "__main__":
    main()
