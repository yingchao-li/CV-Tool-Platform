"""End-to-end box measurement from a depth-camera capture.

The pipeline is deliberately staged so each step can be inspected on its
own (and reused by the interactive studio):

1. keep pixels with a valid depth return;
2. fit the dominant plane -> **floor**;
3. clean the floor mask morphologically and take its complement;
4. fit a second plane on the non-floor points -> **box top**;
5. keep the largest connected component of the box top;
6. measure height from the two planes and length/width from the top
   surface (PCA of the projected points).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dataset import load_example, valid_points_with_indices
from .masks import (
    create_mask_from_indices,
    filter_floor_mask,
    get_non_floor_mask,
    get_points_from_mask,
    largest_connected_component,
)
from .measurement import estimate_box_dimensions, get_box_top_corners_pca
from .mlesac import mlesac_plane
from .planes import Plane
from .preemptive import preemptive_ransac_plane
from .ransac import plane_rmse, ransac_plane


@dataclass
class BoxMeasurement:
    """Result bundle returned by :func:`measure_box`."""

    method: str
    height: float
    length: float
    width: float
    corners: np.ndarray
    floor_plane: Plane
    box_plane: Plane
    floor_inliers: np.ndarray
    box_inliers: np.ndarray
    floor_score: float
    box_score: float
    floor_rmse: float
    box_rmse: float
    floor_mask: np.ndarray
    box_top_mask: np.ndarray
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Serialisable summary, suitable for JSON output."""
        return {
            "method": self.method,
            "height_m": round(float(self.height), 4),
            "length_m": round(float(self.length), 4),
            "width_m": round(float(self.width), 4),
            "floor_score": float(self.floor_score),
            "box_score": float(self.box_score),
            "floor_rmse": round(float(self.floor_rmse), 6),
            "box_rmse": round(float(self.box_rmse), 6),
            "floor_inliers": int(self.floor_inliers.sum()),
            "box_inliers": int(self.box_inliers.sum()),
            "corners": np.asarray(self.corners, dtype=float).round(4).tolist(),
        }


def _fit(
    points: np.ndarray,
    method: str,
    threshold: float,
    max_iterations: int,
    random_seed: int,
    mlesac_gamma: float,
    preemptive_hypotheses: int,
    preemptive_batch: int,
) -> tuple[Plane, np.ndarray, float]:
    if method == "ransac":
        return ransac_plane(
            points,
            threshold=threshold,
            max_iterations=max_iterations,
            random_seed=random_seed,
        )
    if method == "mlesac":
        return mlesac_plane(
            points,
            num_iterations=max_iterations,
            epsilon=threshold,
            gamma=mlesac_gamma,
            random_seed=random_seed,
        )
    if method == "preemptive":
        return preemptive_ransac_plane(
            points,
            num_hypotheses=preemptive_hypotheses,
            batch_size=preemptive_batch,
            threshold=threshold,
            random_seed=random_seed,
        )
    raise ValueError(
        f"Unknown fitting method {method!r}; expected ransac, mlesac or preemptive."
    )


def measure_box(
    cloud: np.ndarray,
    method: str = "ransac",
    floor_threshold: float = 0.05,
    box_threshold: float = 0.005,
    max_iterations: int = 1000,
    random_seed: int = 42,
    mlesac_gamma: float = 0.1,
    preemptive_hypotheses: int = 200,
    preemptive_batch: int = 100,
    min_floor_size: int = 100,
    max_hole_size: int = 300,
    size_method: str = "pca",
) -> BoxMeasurement:
    """Measure the box in ``cloud`` (``H x W x 3``) with the chosen estimator."""
    points, rows, cols = valid_points_with_indices(cloud)
    if points.shape[0] < 3:
        raise ValueError("The capture contains too few valid 3D points.")

    floor_plane, floor_inliers, floor_score = _fit(
        points,
        method,
        floor_threshold,
        max_iterations,
        random_seed,
        mlesac_gamma,
        preemptive_hypotheses,
        preemptive_batch,
    )

    raw_floor_mask = create_mask_from_indices(
        image_shape=cloud.shape[:2],
        rows=rows,
        cols=cols,
        selected_mask=floor_inliers,
    )
    floor_mask = filter_floor_mask(
        raw_floor_mask,
        min_floor_size=min_floor_size,
        max_hole_size=max_hole_size,
    )

    non_floor_mask = get_non_floor_mask(floor_mask)
    non_floor_points, non_floor_rows, non_floor_cols = get_points_from_mask(
        cloud, non_floor_mask
    )
    if non_floor_points.shape[0] < 3:
        raise ValueError("Not enough non-floor points to fit the box top.")

    box_plane, box_inliers, box_score = _fit(
        non_floor_points,
        method,
        box_threshold,
        max_iterations,
        random_seed,
        mlesac_gamma,
        preemptive_hypotheses,
        preemptive_batch,
    )

    raw_box_mask = create_mask_from_indices(
        image_shape=cloud.shape[:2],
        rows=non_floor_rows,
        cols=non_floor_cols,
        selected_mask=box_inliers,
    )
    box_top_mask = largest_connected_component(raw_box_mask)

    height, length, width = estimate_box_dimensions(
        cloud,
        floor_plane,
        box_plane,
        box_top_mask,
        method=size_method,
    )
    corners = get_box_top_corners_pca(cloud, box_top_mask)

    return BoxMeasurement(
        method=method,
        height=float(height),
        length=float(length),
        width=float(width),
        corners=corners,
        floor_plane=floor_plane,
        box_plane=box_plane,
        floor_inliers=floor_inliers,
        box_inliers=box_inliers,
        floor_score=float(floor_score),
        box_score=float(box_score),
        floor_rmse=plane_rmse(points, floor_plane, floor_inliers),
        box_rmse=plane_rmse(non_floor_points, box_plane, box_inliers),
        floor_mask=floor_mask,
        box_top_mask=box_top_mask,
        extra={
            "valid_points": int(points.shape[0]),
            "non_floor_points": int(non_floor_points.shape[0]),
            "floor_threshold": floor_threshold,
            "box_threshold": box_threshold,
        },
    )


def measure_example(example_id: int = 1, **kwargs) -> BoxMeasurement:
    """Convenience wrapper: load ``example{id}kinect.mat`` and measure it."""
    _, _, cloud = load_example(example_id)
    return measure_box(cloud, **kwargs)
