"""Shared plane primitives.

A plane is represented as ``(normal, offset)`` and satisfies::

    normal . x = offset

``normal`` is a unit vector, so ``|normal . x - offset|`` is the Euclidean
point-to-plane distance.  Every robust estimator in :mod:`cvkit.geometry`
returns this same representation, which keeps the downstream measurement
code independent of the fitting strategy.
"""

from __future__ import annotations

import numpy as np

Plane = tuple[np.ndarray, float]


def fit_plane(points: np.ndarray) -> Plane | None:
    """Fit a plane to at least three points with a least-squares normal.

    For exactly three points the plane is defined by their cross product.
    For more points the normal is the smallest principal direction of the
    centred sample (SVD), which is the total-least-squares solution.

    Returns ``None`` when the points are degenerate, collinear or contain
    non-finite values.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape N x 3, got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane.")
    if not np.all(np.isfinite(points)):
        return None

    if points.shape[0] == 3:
        p1, p2, p3 = points
        with np.errstate(over="ignore", invalid="ignore"):
            normal = np.cross(p2 - p1, p3 - p1)
    else:
        centroid = np.mean(points, axis=0)
        try:
            _, singular_values, vt = np.linalg.svd(
                points - centroid, full_matrices=False
            )
        except np.linalg.LinAlgError:
            return None
        if singular_values.shape[0] < 3 or singular_values[-2] < 1e-12:
            return None
        normal = vt[-1]

    if not np.all(np.isfinite(normal)):
        return None

    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm < 1e-12:
        return None

    normal = normal / norm
    offset = float(np.dot(normal, np.mean(points, axis=0)))

    if not np.all(np.isfinite(normal)) or not np.isfinite(offset):
        return None

    return normal, offset


def point_plane_distances(points: np.ndarray, plane: Plane) -> np.ndarray:
    """Return the absolute distance from every point to ``plane``.

    Non-finite points get ``inf`` so they can never be counted as inliers.
    """
    points = np.asarray(points, dtype=float)
    normal, offset = plane
    normal = np.asarray(normal, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape N x 3, got {points.shape}")
    if normal.shape != (3,):
        raise ValueError(f"normal must have shape (3,), got {normal.shape}")

    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm < 1e-12:
        raise ValueError("Plane normal must not be zero.")

    finite = np.all(np.isfinite(points), axis=1)
    distances = np.full(points.shape[0], np.inf, dtype=float)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        values = np.abs(points[finite] @ normal - offset) / normal_norm

    values[~np.isfinite(values)] = np.inf
    distances[finite] = values
    return distances


def plane_inliers(points: np.ndarray, plane: Plane, threshold: float) -> np.ndarray:
    """Boolean inlier mask for ``threshold`` (distance below threshold)."""
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    return point_plane_distances(points, plane) < threshold
