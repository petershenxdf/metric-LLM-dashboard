"""Tests for the SSDBCODI algorithm."""
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from app.domain.clustering.ssdbcodi import SSDBCODI
from app.domain.clustering.distance import make_distance


def test_cold_start_runs_dbscan():
    """With no DN labels, the cold-start path should run DBSCAN and discover
    the underlying cluster structure instead of collapsing everything into one
    cluster."""
    X, _ = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.5,
        random_state=42,
    )
    algo = SSDBCODI(min_pts=5, k_outliers=2)
    result = algo.fit(X, DN={}, DO=set(), distance_func=make_distance(n_features=2))

    assert result.cluster_labels.shape == (150,)
    assert result.is_outlier.shape == (150,)

    # DBSCAN should have discovered at least 2 distinct clusters on blobs data
    valid_clusters = result.cluster_labels[result.cluster_labels >= 0]
    n_clusters = len(np.unique(valid_clusters))
    assert n_clusters >= 2, f"Expected DBSCAN to find >=2 clusters, got {n_clusters}"


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


def test_single_class_must_link_does_not_collapse(moons_dataset):
    """Regression: giving the algorithm a single must-link group should NOT
    collapse every point into that one cluster. Instead it should fall back
    to unsupervised DBSCAN for the rest and overlay the labeled group as its
    own cluster."""
    X = moons_dataset
    # Label 20 points as all sharing cluster id 5 (i.e. a must-link group)
    DN = {i: 5 for i in range(20)}
    DO = set()

    algo = SSDBCODI(min_pts=3, k_outliers=3)
    md = make_distance(n_features=2)
    result = algo.fit(X, DN, DO, distance_func=md)

    # The labeled group must be in their assigned cluster
    for i in DN.keys():
        assert result.cluster_labels[i] == 5

    # There must be more than one cluster in the final output -- otherwise we
    # collapsed (the bug we're guarding against).
    unique_clusters = set(int(l) for l in result.cluster_labels if l >= 0)
    assert len(unique_clusters) >= 2, (
        f"Expected the single-class fallback to keep >=2 clusters, got {unique_clusters}"
    )


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
