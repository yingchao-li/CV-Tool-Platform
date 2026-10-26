"""MLESAC plane fitting.

MLESAC (Torr & Zisserman, 2000) keeps the RANSAC sampling strategy but
replaces the inlier-count objective with a robust likelihood cost::

    cost = d_i          if d_i < epsilon
    cost = gamma        if d_i >= epsilon

A larger ``gamma`` makes outliers cheaper relative to inliers, so a model
with slightly fewer inliers but much smaller residuals can win.  This is
what lets MLESAC return geometrically tighter planes than vanilla RANSAC.
"""

from __future__ import annotations

import numpy as np

from .planes import Plane, fit_plane, point_plane_distances


def mlesac_plane(
    points: np.ndarray,
    num_iterations: int = 1000,
    epsilon: float = 0.05,
    gamma: float = 0.1,
    random_seed: int = 42,
) -> tuple[Plane, np.ndarray, float]:
    """Fit the dominant plane by minimising the MLESAC cost.

    Parameters
    ----------
    points:
        ``(N, 3)`` array of 3D points.
    num_iterations:
        Number of random hypotheses.
    epsilon:
        Inlier residual threshold.
    gamma:
        Constant cost assigned to every outlier.  Must be ``> epsilon``.
    random_seed:
        Seed for reproducible sampling.

    Returns
    -------
    plane, inliers, cost
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape N x 3, got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if gamma <= epsilon:
        raise ValueError("gamma should be larger than epsilon.")

    finite_indices = np.where(np.all(np.isfinite(points), axis=1))[0]
    if finite_indices.size < 3:
        raise ValueError("Need at least 3 finite points to fit a plane.")

    rng = np.random.default_rng(random_seed)
    n_points = points.shape[0]
    best_plane: Plane | None = None
    best_inliers = np.zeros(n_points, dtype=bool)
    best_cost = np.inf

    for _ in range(num_iterations):
        sample = points[rng.choice(finite_indices, 3, replace=False)]
        plane = fit_plane(sample)
        if plane is None:
            continue

        distances = point_plane_distances(points, plane)
        cost = float(np.sum(np.where(distances < epsilon, distances, gamma)))

        if cost < best_cost:
            best_cost = cost
            best_plane = plane
            best_inliers = distances < epsilon

    if best_plane is None:
        raise RuntimeError("MLESAC failed to find a valid plane.")

    return best_plane, best_inliers, best_cost
