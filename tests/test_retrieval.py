"""Unit tests for VLAD encoding, normalisation and retrieval metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cvkit.retrieval.pipeline import (
    assignments,
    evaluate,
    nearest_cluster_indices,
    normalize_vlad,
    raw_vlad,
)


def test_assignments_are_one_hot():
    descriptors = np.array([[0.0, 0.0], [10.0, 10.0], [0.1, 0.0]])
    clusters = np.array([[0.0, 0.0], [10.0, 10.0]])

    matrix = assignments(descriptors, clusters)
    assert matrix.shape == (3, 2)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0


def test_nearest_cluster_indices():
    descriptors = np.array([[0.0, 0.0], [10.0, 10.0]])
    clusters = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert nearest_cluster_indices(descriptors, clusters).tolist() == [0, 1]


def test_normalize_vlad_is_l2_normalised():
    vector = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)
    normalised = normalize_vlad(vector, powernorm=False)
    assert np.linalg.norm(normalised) == pytest.approx(1.0)


def test_power_norm_compresses_large_components():
    vector = np.array([9.0, 1.0], dtype=np.float32)
    plain = normalize_vlad(vector, powernorm=False)
    powered = normalize_vlad(vector, powernorm=True)
    # The ratio between components shrinks after signed square-root.
    assert (powered[0] / powered[1]) < (plain[0] / plain[1])


def test_vlad_dimension_matches_codebook():
    rng = np.random.default_rng(0)
    descriptors = np.vstack(
        [
            rng.normal(0.0, 0.1, (10, 8)),
            rng.normal(1.0, 0.1, (10, 8)),
        ]
    )
    clusters = descriptors[[0, 15]]
    raw = raw_vlad(descriptors, clusters)
    normalised = normalize_vlad(raw, powernorm=False)

    assert raw.shape == (2 * 8,)
    assert np.linalg.norm(normalised) == pytest.approx(1.0)


def test_evaluate_perfect_retrieval():
    # Two identical encodings per class: self is excluded by the metric, so the
    # nearest neighbour of every query is its twin and retrieval is perfect.
    basis = np.eye(4, dtype=np.float32)
    encodings = np.repeat(basis, 2, axis=0)
    top1, mean_ap = evaluate(encodings, [0, 0, 1, 1, 2, 2, 3, 3])

    assert top1 == pytest.approx(1.0)
    assert mean_ap == pytest.approx(1.0)
