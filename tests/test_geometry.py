"""Unit tests for the geometry primitives and the three robust estimators."""

from __future__ import annotations

import numpy as np

from cvkit.geometry.mlesac import mlesac_plane
from cvkit.geometry.planes import fit_plane, plane_inliers, point_plane_distances
from cvkit.geometry.preemptive import preemptive_ransac_plane
from cvkit.geometry.ransac import plane_rmse, ransac_plane


def _plane_with_outliers(n_inliers=400, offset=2.0, outliers=60, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, size=(n_inliers, 2))
    z = offset + rng.normal(0.0, 0.001, size=n_inliers)
    inliers = np.column_stack([xy, z])
    noise = rng.uniform(-2.0, 2.0, size=(outliers, 3))
    return np.vstack([inliers, noise]), n_inliers


def test_fit_plane_recovers_axis_aligned_plane():
    rng = np.random.default_rng(1)
    points = np.column_stack(
        [rng.uniform(-1, 1, (50, 2)), np.full(50, 2.0)]
    )
    plane = fit_plane(points)

    assert plane is not None
    normal, offset = plane
    assert abs(abs(normal[2]) - 1.0) < 1e-9
    assert abs(offset - 2.0) < 1e-9

    distances = point_plane_distances(np.array([[0.0, 0.0, 2.0]]), plane)
    assert distances[0] < 1e-9


def test_fit_plane_rejects_collinear_points():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert fit_plane(points) is None


def test_ransac_recovers_plane_with_outliers():
    points, n_inliers = _plane_with_outliers()
    plane, inliers, score = ransac_plane(points, threshold=0.01, max_iterations=300)

    assert inliers.sum() >= n_inliers
    assert score == points.shape[0] - int(inliers.sum())
    assert abs(plane[1] - 2.0) < 0.01
    assert plane_rmse(points, plane, inliers) < 0.01


def test_mlesac_prefers_small_residuals():
    points, _ = _plane_with_outliers()
    plane, inliers, cost = mlesac_plane(
        points, num_iterations=300, epsilon=0.01, gamma=0.1
    )

    assert cost >= 0.0
    assert abs(plane[1] - 2.0) < 0.01
    assert plane_inliers(points, plane, 0.01).sum() >= inliers.sum()


def test_preemptive_matches_ransac_within_tolerance():
    points, _ = _plane_with_outliers()
    plane, inliers, cost = preemptive_ransac_plane(
        points, num_hypotheses=100, batch_size=50, threshold=0.01
    )

    assert cost >= 0
    assert inliers.mean() > 0.8
    assert abs(plane[1] - 2.0) < 0.02
