"""Vanilla RANSAC plane fitting.

The objective is the inlier count: each hypothesis is sampled from three
points and the model that explains the largest number of points within
``threshold`` wins.  This is the baseline against which MLESAC and
Preemptive RANSAC are compared.
"""

from __future__ import annotations

import numpy as np

from .planes import Plane, fit_plane, plane_inliers, point_plane_distances


def ransac_plane(
    points: np.ndarray,
    threshold: float = 0.01,
    max_iterations: int = 1000,
    random_seed: int = 42,
) -> tuple[Plane, np.ndarray, int]:
    """Fit the dominant plane with RANSAC.

    Parameters
    ----------
    points:
        ``(N, 3)`` array of 3D points.
    threshold:
        Inlier distance threshold, in the same units as the point cloud.
    max_iterations:
        Number of random three-point hypotheses to evaluate.
    random_seed:
        Seed for reproducible sampling.

    Returns
    -------
    plane:
        ``(normal, offset)`` with ``normal . x = offset``.
    inliers:
        Boolean mask of shape ``(N,)``.
    score:
        Number of outliers (lower is better).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape N x 3, got {points.shape}")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    n_points = points.shape[0]
    if n_points < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    finite_indices = np.where(np.all(np.isfinite(points), axis=1))[0]
    if finite_indices.size < 3:
        raise ValueError("Need at least 3 finite points to fit a plane.")

    rng = np.random.default_rng(random_seed)
    best_plane: Plane | None = None
    best_inliers = np.zeros(n_points, dtype=bool)
    best_count = 0

    for _ in range(max_iterations):
        sample = points[rng.choice(finite_indices, 3, replace=False)]
        plane = fit_plane(sample)
        if plane is None:
            continue

        inliers = plane_inliers(points, plane, threshold)
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_plane = plane
            best_inliers = inliers
            if best_count == n_points:
                break

    if best_plane is None:
        raise RuntimeError("RANSAC failed to find a valid plane.")

    return best_plane, best_inliers, n_points - best_count


def plane_rmse(
    points: np.ndarray, plane: Plane, inliers: np.ndarray | None = None
) -> float:
    """Root-mean-square point-to-plane distance over the inliers."""
    distances = point_plane_distances(points, plane)
    if inliers is not None:
        distances = distances[inliers]
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))
