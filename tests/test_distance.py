"""Tests for the distance abstractions."""
import numpy as np
import pytest

from app.domain.clustering.distance import (
    MahalanobisDistance,
    euclidean_distance,
    make_distance,
)


def test_identity_mahalanobis_equals_euclidean():
    """With M = I, Mahalanobis distance should equal Euclidean."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))
    md = make_distance(n_features=3)

    for i in range(len(X)):
        for j in range(len(X)):
            expected = euclidean_distance(X[i], X[j])
            actual = md(X[i], X[j])
            assert abs(expected - actual) < 1e-10


def test_pairwise_shape():
    X = np.random.rand(20, 5)
    md = make_distance(n_features=5)
    D = md.pairwise(X)
    assert D.shape == (20, 20)
    # Distance matrix should be symmetric
    assert np.allclose(D, D.T, atol=1e-8)
    # Diagonal should be ~0
    assert np.allclose(np.diag(D), 0.0, atol=1e-8)


def test_update_M_stays_psd():
    """After updating with a noisy M, the distance should still be valid."""
    md = make_distance(n_features=3)
    # A matrix with small negative eigenvalue -- should be corrected
    noisy_M = np.array([
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.0, -0.01],  # Small negative
    ])
    md.update_M(noisy_M)

    # Every self-distance should be non-negative and finite
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.5])
    d = md(x, y)
    assert d >= 0
    assert np.isfinite(d)
