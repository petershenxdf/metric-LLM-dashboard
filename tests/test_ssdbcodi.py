"""Tests for the SSDBCODI algorithm."""
import numpy as np
import pytest

from app.domain.clustering.ssdbcodi import SSDBCODI
from app.domain.clustering.distance import make_distance


def test_cold_start_returns_single_cluster():
    """With no DN labels, the fallback path should put everything in cluster 0."""
    X = np.random.rand(20, 3)
    algo = SSDBCODI(min_pts=3, k_outliers=2)
    result = algo.fit(X, DN={}, DO=set(), distance_func=make_distance(n_features=3))

    assert result.cluster_labels.shape == (20,)
    assert result.is_outlier.shape == (20,)
    # All non-outlier points should be cluster 0
    assert np.all(result.cluster_labels == 0)


def test_fit_with_labels_returns_expected_shapes(moons_dataset):
    """Full fit on the moons dataset -- verify output shapes and sanity."""
    X = moons_dataset
    n = len(X)

    # Label 3 points in each moon -- easy because make_moons is deterministic
    # Points 0, 1, 2 are in one moon; points 50, 51, 52 are in the other
    DN = {0: 0, 1: 0, 2: 0, 50: 1, 51: 1, 52: 1}
    DO = set()

    algo = SSDBCODI(min_pts=3, alpha=0.4, beta=0.4, k_outliers=5)
    md = make_distance(n_features=2)
    result = algo.fit(X, DN, DO, distance_func=md)

    # Shapes
    assert result.cluster_labels.shape == (n,)
    assert result.is_outlier.shape == (n,)
    assert result.rscore.shape == (n,)
    assert result.lscore.shape == (n,)
    assert result.simscore.shape == (n,)
    assert result.tscore.shape == (n,)

    # Score ranges
    assert np.all(result.rscore >= 0) and np.all(result.rscore <= 1)
    assert np.all(result.lscore >= 0) and np.all(result.lscore <= 1)

    # User-labeled points should keep their labels in the final result
    for idx, cid in DN.items():
        assert result.cluster_labels[idx] == cid


def test_labeled_outliers_remain_outliers(moons_dataset):
    X = moons_dataset
    DN = {0: 0, 1: 0, 50: 1, 51: 1}
    DO = {10, 20}

    algo = SSDBCODI(min_pts=3, k_outliers=3)
    md = make_distance(n_features=2)
    result = algo.fit(X, DN, DO, distance_func=md)

    for idx in DO:
        assert result.is_outlier[idx] == True
        assert result.cluster_labels[idx] == -1
