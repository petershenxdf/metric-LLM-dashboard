"""Tests for the composite metric learner."""
import numpy as np

from app.domain.metric_learning.composite import CompositeMetricLearner
from app.domain.constraints.schemas import (
    MustLink,
    CannotLink,
    Triplet,
    FeatureHint,
)


def test_composite_starts_at_identity():
    learner = CompositeMetricLearner(n_features=4)
    M = learner.get_M()
    assert np.allclose(M, np.eye(4))


def test_reset_restores_identity():
    learner = CompositeMetricLearner(n_features=3)
    # Perturb M manually
    learner.M = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 2.0]])
    learner.reset(3)
    assert np.allclose(learner.get_M(), np.eye(3))


def test_triplet_update_changes_M():
    """A triplet update should modify M in the expected direction."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))
    learner = CompositeMetricLearner(n_features=3, triplet_lr=0.1)

    M_before = learner.get_M().copy()

    # Construct a triplet where anchor is far from positive and close to negative
    # -- this is a margin violation, so the update should fire
    triplet = Triplet(anchor=0, positive=5, negative=1)
    learner.update(X, constraint=triplet)

    M_after = learner.get_M()
    # M should have changed (triplet SGD kicked in)
    assert not np.allclose(M_before, M_after)
    # M should still be positive definite
    eigvals = np.linalg.eigvalsh(M_after)
    assert np.all(eigvals >= -1e-6)


def test_feature_hint_decrease_shrinks_diagonal():
    learner = CompositeMetricLearner(
        n_features=3,
        feature_names=["height", "weight", "age"],
    )
    before = learner.get_M()[1, 1]

    hint = FeatureHint(feature_name="weight", direction="decrease", magnitude="moderate")
    learner.update(X=np.zeros((1, 3)), constraint=hint)

    after = learner.get_M()[1, 1]
    assert after < before


def test_feature_hint_ignore_zeros_feature():
    learner = CompositeMetricLearner(
        n_features=3,
        feature_names=["a", "b", "c"],
    )
    hint = FeatureHint(feature_name="b", direction="ignore", magnitude="strong")
    learner.update(X=np.zeros((1, 3)), constraint=hint)

    M = learner.get_M()
    # Row and column for feature b should be near zero
    assert M[1, 1] < 1e-3
    # Matrix should still be symmetric
    assert np.allclose(M, M.T)


def test_must_link_runs_itml_without_crashing():
    """ITML needs both similar and dissimilar pairs -- this test just verifies
    that a must_link-only update does not crash and keeps M sane."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4))
    learner = CompositeMetricLearner(n_features=4)

    ml = MustLink(point_ids=[0, 1, 2])
    learner.update(X, constraint=ml)

    M = learner.get_M()
    # Still PSD
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals >= -1e-6)
