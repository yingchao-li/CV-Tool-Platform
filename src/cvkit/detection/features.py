"""Fixed-length HOG features for proposal crops."""

from dataclasses import asdict, dataclass

import numpy as np
import skimage.feature
import skimage.transform
import skimage.util

from .boxes import clip_box, is_valid_box


@dataclass(frozen=True)
class FeatureConfig:
    resize_height: int = 64
    resize_width: int = 64
    orientations: int = 9
    pixels_per_cell: int = 8
    cells_per_block: int = 2
    min_crop_size: int = 4

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


def extract_feature(image, box, config=None):
    """Extract one HOG vector, or return None for an invalid crop."""
    config = config or FeatureConfig()
    height, width = image.shape[:2]
    box = clip_box(box, width, height)
    if not is_valid_box(box, config.min_crop_size):
        return None

    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < config.min_crop_size or crop.shape[1] < config.min_crop_size:
        return None

    resized = skimage.transform.resize(
        crop,
        (config.resize_height, config.resize_width),
        preserve_range=True,
        anti_aliasing=True,
    )
    resized = skimage.util.img_as_float32(resized)
    feature = skimage.feature.hog(
        resized,
        orientations=config.orientations,
        pixels_per_cell=(config.pixels_per_cell, config.pixels_per_cell),
        cells_per_block=(config.cells_per_block, config.cells_per_block),
        block_norm="L2-Hys",
        channel_axis=-1,
        feature_vector=True,
    )
    return np.asarray(feature, dtype=np.float32)


def extract_features(image, boxes, config=None):
    """Extract valid features and return (features, boxes, original indices)."""
    config = config or FeatureConfig()
    features = []
    valid_boxes = []
    valid_indices = []
    for index, box in enumerate(np.asarray(boxes).reshape(-1, 4)):
        feature = extract_feature(image, box, config)
        if feature is None:
            continue
        features.append(feature)
        valid_boxes.append(box)
        valid_indices.append(index)

    if not features:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.vstack(features).astype(np.float32, copy=False),
        np.asarray(valid_boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(valid_indices, dtype=np.int64),
    )
