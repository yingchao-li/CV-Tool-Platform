"""Point-cloud box measurement and robust plane fitting."""

from __future__ import annotations

from .dataset import (
    describe_data,
    list_mat_files,
    load_example,
    load_mat_file,
    valid_point_mask,
    valid_points,
    valid_points_with_indices,
)
from .masks import (
    create_mask_from_indices,
    filter_floor_mask,
    get_non_floor_mask,
    get_points_from_mask,
    largest_connected_component,
)
from .measurement import (
    estimate_box_dimensions,
    get_box_top_corners_pca,
    plane_distance,
)
from .mlesac import mlesac_plane
from .planes import fit_plane, plane_inliers, point_plane_distances
from .preemptive import preemptive_ransac_plane
from .ransac import plane_rmse, ransac_plane

METHODS = ("ransac", "mlesac", "preemptive")

__all__ = [
    "METHODS",
    "create_mask_from_indices",
    "describe_data",
    "estimate_box_dimensions",
    "filter_floor_mask",
    "fit_plane",
    "get_box_top_corners_pca",
    "get_non_floor_mask",
    "get_points_from_mask",
    "largest_connected_component",
    "list_mat_files",
    "load_example",
    "load_mat_file",
    "mlesac_plane",
    "plane_distance",
    "plane_inliers",
    "plane_rmse",
    "point_plane_distances",
    "preemptive_ransac_plane",
    "ransac_plane",
    "valid_point_mask",
    "valid_points",
    "valid_points_with_indices",
]
