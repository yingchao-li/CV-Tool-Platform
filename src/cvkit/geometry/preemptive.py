"""Preemptive RANSAC plane fitting.

Preemptive RANSAC (Nister, 2003) trades a little accuracy for a large
speed-up.  Instead of scoring one hypothesis at a time against every
point, it generates ``M`` hypotheses up front, then streams the data in
batches of ``B`` points.  After every batch the hypotheses are ranked by
their accumulated outlier count and the worse half is discarded.

Because most bad hypotheses are removed after seeing only a few hundred
points, the cost collapses from ``O(M * N)`` to roughly ``O(M * N / B)``.
"""

from __future__ import annotations

import numpy as np

from .planes import Plane, fit_plane, point_plane_distances


def generate_hypotheses(
    points: np.ndarray,
    num_hypotheses: int = 200,
    sample_size: int = 3,
    random_seed: int = 42,
) -> list[dict]:
    """Sample ``num_hypotheses`` candidate planes from the point cloud."""
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape N x 3, got {points.shape}")
    if num_hypotheses <= 0:
        raise ValueError("num_hypotheses must be positive.")
    if sample_size < 3:
        raise ValueError("sample_size must be at least 3 for plane fitting.")

    finite_indices = np.where(np.all(np.isfinite(points), axis=1))[0]
    if finite_indices.size < sample_size:
        raise ValueError("Not enough finite points to generate hypotheses.")

    rng = np.random.default_rng(random_seed)
    hypotheses: list[dict] = []

    for _ in range(num_hypotheses):
        sample = points[rng.choice(finite_indices, size=sample_size, replace=False)]
        model = fit_plane(sample)
        if model is not None:
            hypotheses.append({"model": model, "cost": 0})

    return hypotheses


def preemptive_ransac_plane(
    points: np.ndarray,
    num_hypotheses: int = 200,
    batch_size: int = 100,
    sample_size: int = 3,
    threshold: float = 0.01,
    random_seed: int = 42,
) -> tuple[Plane, np.ndarray, int]:
    """Fit the dominant plane with Preemptive RANSAC.

    Returns
    -------
    plane, inliers, cost
        ``cost`` is the number of outliers accumulated while streaming the
        data, i.e. the score actually optimised by the algorithm.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape N x 3, got {points.shape}")
    if points.shape[0] < sample_size:
        raise ValueError("Not enough points for the requested sample_size.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    hypotheses = generate_hypotheses(
        points,
        num_hypotheses=num_hypotheses,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    if not hypotheses:
        raise RuntimeError("Preemptive RANSAC failed to generate valid hypotheses.")

    rng = np.random.default_rng(random_seed + 1)
    order = rng.permutation(points.shape[0])
    cursor = 0

    while len(hypotheses) > 1 and cursor < points.shape[0]:
        batch = points[order[cursor : cursor + batch_size]]

        for hypothesis in hypotheses:
            distances = point_plane_distances(batch, hypothesis["model"])
            hypothesis["cost"] += int(np.sum(distances >= threshold))

        hypotheses.sort(key=lambda hypothesis: hypothesis["cost"])
        hypotheses = hypotheses[: max(1, len(hypotheses) // 2)]
        cursor += batch_size

    hypotheses.sort(key=lambda hypothesis: hypothesis["cost"])
    best_model = hypotheses[0]["model"]
    best_cost = int(hypotheses[0]["cost"])

    best_inliers = point_plane_distances(points, best_model) < threshold
    return best_model, best_inliers, best_cost
