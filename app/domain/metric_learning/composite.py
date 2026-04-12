"""Composite metric learner — the glue that lets ITML and Triplet SGD share
one M matrix.

The dashboard wants a single object it can call `update(constraint)` on,
without caring which sub-learner handles which constraint type. This class:

1. Owns the canonical M matrix.
2. Holds an ITMLLearner and a TripletLearner, both initialized with the same M.
3. Routes each constraint to the appropriate sub-learner.
4. After each update, copies the updated M back into the other sub-learner so
   they stay in sync. ITML and triplet operate sequentially on the same M.
5. Handles `feature_hint` constraints directly (modifying the diagonal of M).

This is the dependency-inversion entry point for metric learning. Any new
metric learner type only has to be plugged in here — no other code changes.
"""
import numpy as np
from typing import List, Tuple, Set

from .base import MetricLearner
from .itml_learner import ITMLLearner
from .triplet_learner import TripletLearner

from app.domain.constraints.schemas import (
    Constraint,
    MustLink,
    CannotLink,
    Triplet as TripletConstraint,
    FeatureHint,
    ClusterMerge,
    Reassign,
)


class CompositeMetricLearner(MetricLearner):
    def __init__(
        self,
        n_features: int,
        feature_names: List[str] = None,
        itml_gamma: float = 1.0,
        triplet_lr: float = 0.01,
    ):
        self.n_features = n_features
        self.feature_names = feature_names or [f"f{i}" for i in range(n_features)]

        # Both sub-learners start at identity (Euclidean)
        self.itml = ITMLLearner(n_features=n_features, gamma=itml_gamma)
        self.triplet = TripletLearner(n_features=n_features, lr=triplet_lr)

        self.M: np.ndarray = np.eye(n_features)

    def reset(self, n_features: int = None) -> None:
        n = n_features or self.n_features
        self.n_features = n
        self.M = np.eye(n)
        self.itml.reset(n)
        self.triplet.reset(n)

    def get_M(self) -> np.ndarray:
        return self.M.copy()

    def _sync_M(self):
        """Copy the canonical M into both sub-learners after any update."""
        self.itml.M = self.M.copy()
        self.triplet.set_M(self.M)

    def update(self, X: np.ndarray, constraint: Constraint = None, **kwargs) -> None:
        """Route a constraint to the appropriate sub-learner.

        The signature is intentionally permissive (kwargs) to satisfy the
        abstract base class. The dashboard always calls with a Constraint
        object.
        """
        if constraint is None:
            return

        if isinstance(constraint, MustLink):
            self._handle_must_link(X, constraint)
        elif isinstance(constraint, CannotLink):
            self._handle_cannot_link(X, constraint)
        elif isinstance(constraint, TripletConstraint):
            self._handle_triplet(X, constraint)
        elif isinstance(constraint, FeatureHint):
            self._handle_feature_hint(constraint)
        elif isinstance(constraint, ClusterMerge):
            # Cluster merge alone has no metric implication; the label channel
            # handles it. We do nothing here.
            pass
        elif isinstance(constraint, Reassign):
            # Reassign is also primarily a label-channel operation, but it
            # also implies a relative-similarity hint. We skip metric updates
            # here for simplicity — the label change is enough to drive
            # re-clustering.
            pass
        else:
            # Other constraint types (cluster_count, outlier_label) don't
            # affect the metric.
            pass

        self._sync_M()

    def _handle_must_link(self, X: np.ndarray, c: MustLink) -> None:
        """Apply a must-link constraint to the metric.

        ITML needs BOTH similar and dissimilar pairs to fit — on the first
        must-link (no cannot-links yet) it would silently skip the update and
        M would stay at identity. That made metric learning invisible to the
        user. To fix this we also do a direct whitening update: compute the
        within-group covariance of the must-link set and blend its inverse
        into M. Directions along which the must-link points vary become
        *less* important, shrinking within-group distances — which is exactly
        what "these are the same class" should mean geometrically.

        We still feed the pairs to ITML so that once a dissimilar pair shows
        up later, ITML can refine using the full accumulated history.
        """
        ids = c.point_ids
        if len(ids) < 2:
            return

        # 1) Direct whitening update on the shared M
        group_X = X[ids]
        center = group_X.mean(axis=0)
        centered = group_X - center
        within_cov = (centered.T @ centered) / max(len(ids) - 1, 1)

        # Regularize with a fraction of the mean diagonal so the inverse is
        # well-conditioned even if the group is collinear in some directions.
        mean_diag = float(np.trace(within_cov) / self.n_features)
        if mean_diag <= 0:
            mean_diag = 1.0
        reg = 1e-2 * mean_diag * np.eye(self.n_features)
        try:
            M_whiten = np.linalg.inv(within_cov + reg)
        except np.linalg.LinAlgError:
            M_whiten = np.linalg.pinv(within_cov + reg)

        # Normalize M_whiten so its trace matches current M's trace — this
        # keeps the scale of distances roughly comparable across updates.
        cur_trace = float(np.trace(self.M))
        whiten_trace = float(np.trace(M_whiten))
        if whiten_trace > 0:
            M_whiten = M_whiten * (cur_trace / whiten_trace)

        # Blend: mostly the new whitening, a little of the existing metric
        blend_alpha = 0.7
        blended = blend_alpha * M_whiten + (1.0 - blend_alpha) * self.M
        self.M = TripletLearner._project_to_psd(blended)

        # 2) Also accumulate pairs into ITML for future joint refinement
        pairs = []
        labels = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))
                labels.append(1)
        # ITML's shared M prior should reflect what we just computed
        self.itml.M = self.M.copy()
        self.itml.update(X, pairs=pairs, labels=labels)
        # If ITML actually ran (both classes present), take its refined M
        if len(set(self.itml.labels)) >= 2:
            self.M = self.itml.get_M()

    def _handle_cannot_link(self, X: np.ndarray, c: CannotLink) -> None:
        """Cross product of group_a and group_b becomes dissimilar pairs."""
        if not c.group_a or not c.group_b:
            return
        pairs = []
        labels = []
        for i in c.group_a:
            for j in c.group_b:
                pairs.append((i, j))
                labels.append(-1)
        self.itml.update(X, pairs=pairs, labels=labels)
        self.M = self.itml.get_M()

    def _handle_triplet(self, X: np.ndarray, c: TripletConstraint) -> None:
        """Single SGD step on the triplet learner, then copy M back."""
        self.triplet.update(
            X,
            anchor=c.anchor,
            positive=c.positive,
            negative=c.negative,
        )
        self.M = self.triplet.get_M()

    def _handle_feature_hint(self, c: FeatureHint) -> None:
        """Directly scale the corresponding diagonal entry of M.

        Magnitude controls how aggressively we scale:
            slight   -> 0.7
            moderate -> 0.4
            strong   -> 0.1
        For 'ignore', we zero out the row+column for that feature.
        """
        # Find the feature index
        feature_idx = None
        if c.feature_name in self.feature_names:
            feature_idx = self.feature_names.index(c.feature_name)
        else:
            # Try a case-insensitive substring match
            lname = c.feature_name.lower()
            for i, name in enumerate(self.feature_names):
                if lname in name.lower() or name.lower() in lname:
                    feature_idx = i
                    break
        if feature_idx is None:
            return

        scale_map = {"slight": 0.7, "moderate": 0.4, "strong": 0.1}
        scale = scale_map.get(c.magnitude, 0.4)

        if c.direction == "ignore":
            # Zero out the row and column, leave a tiny diagonal entry to keep
            # M positive definite.
            self.M[feature_idx, :] = 0.0
            self.M[:, feature_idx] = 0.0
            self.M[feature_idx, feature_idx] = 1e-8
        elif c.direction == "decrease":
            self.M[feature_idx, feature_idx] *= scale
        elif c.direction == "increase":
            # Increase = inverse of the scale (1/0.7, 1/0.4, etc.)
            self.M[feature_idx, feature_idx] /= max(scale, 1e-6)

        # Re-symmetrize and PSD-project to be safe
        self.M = TripletLearner._project_to_psd(self.M)
