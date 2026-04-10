"""SSDBCODI — semi-supervised density-based clustering with outlier detection.

This is Algorithm 2 from the paper, with the modifications needed to plug it
into the dashboard:

1. Distance is supplied via an injected `distance_func` (default Euclidean)
   so metric learning can replace it transparently.
2. After SSDBSCAN expansion, points left unclustered get classified by a
   non-linear classifier trained on the reliable normal/outlier sets, with
   sample weights derived from rScore / tScore.
3. The fit() method returns everything the dashboard needs in one shot:
   cluster labels, outlier flags, and the three scores.

The algorithm has zero I/O and zero dependence on Flask. It can be unit-tested
in isolation.
"""
import numpy as np
from typing import Dict, Set, Optional
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier

from .ssdbscan import ssdbscan
from .scores import (
    compute_reachability_score,
    compute_local_density_score,
    compute_similarity_score,
    compute_total_score,
)
from .distance import MahalanobisDistance, make_distance


@dataclass
class SSDBCODIResult:
    cluster_labels: np.ndarray  # shape (n,) cluster id per point
    is_outlier: np.ndarray      # shape (n,) bool
    rscore: np.ndarray
    lscore: np.ndarray
    simscore: np.ndarray
    tscore: np.ndarray


class SSDBCODI:
    """The full SSDBCODI pipeline as described in the paper.

    Usage:
        algo = SSDBCODI(min_pts=3, alpha=0.4, beta=0.4, k_outliers=10)
        result = algo.fit(X, DN={0: 0, 50: 1, 100: 2}, DO={5, 7}, distance_func=md)
        result.cluster_labels  # ndarray of cluster ids
        result.is_outlier      # ndarray of bools
    """

    def __init__(
        self,
        min_pts: int = 3,
        alpha: float = 0.4,
        beta: float = 0.4,
        k_outliers: int = 10,
        classifier=None,
        random_state: int = 42,
    ):
        self.min_pts = min_pts
        self.alpha = alpha
        self.beta = beta
        self.k_outliers = k_outliers
        self.random_state = random_state
        # Default classifier — random forest handles non-convex shapes well and
        # natively supports sample_weight, which we need for the score-weighted
        # training in step 3 of the algorithm.
        self.classifier = classifier or RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(
        self,
        X: np.ndarray,
        DN: Dict[int, int],
        DO: Set[int],
        distance_func: Optional[MahalanobisDistance] = None,
    ) -> SSDBCODIResult:
        """Run the full algorithm.

        Args:
            X: data matrix (n, d).
            DN: dict point_idx -> cluster_label (labeled normal points).
            DO: set of point_idx labeled as outliers.
            distance_func: distance object with `pairwise(X)` method. Defaults
                to Euclidean (identity Mahalanobis) if not provided.

        Returns:
            SSDBCODIResult containing cluster labels, outlier flags, and scores.
        """
        n = len(X)
        if distance_func is None:
            distance_func = make_distance(n_features=X.shape[1])

        # If we have no labels at all, fall back to a single-cluster output
        # so the dashboard can still show something on the first run.
        if not DN:
            return self._cold_start_fallback(X, DO)

        # Step 1: SSDBSCAN expansion
        cluster_assignments, dist_matrix, rdist_matrix, core_dists = ssdbscan(
            X, DN, distance_func, self.min_pts
        )

        # Step 2: compute the three scores and combine into tScore
        rscore = compute_reachability_score(rdist_matrix, DN)
        lscore = compute_local_density_score(rdist_matrix, self.min_pts)
        simscore = compute_similarity_score(dist_matrix, DO)
        tscore = compute_total_score(rscore, lscore, simscore, self.alpha, self.beta)

        # Step 3: pick reliable normals (RN) and reliable outliers (RO)
        # RN = the points SSDBSCAN successfully clustered (excluding any user-labeled outliers)
        RN_indices = [i for i in cluster_assignments.keys() if i not in DO]
        # RO = the user-labeled outliers plus the top-k highest-tScore unclustered points
        unclustered = [i for i in range(n) if i not in cluster_assignments and i not in DO]
        unclustered_sorted_by_tscore = sorted(unclustered, key=lambda i: -tscore[i])
        top_k_outliers = unclustered_sorted_by_tscore[: self.k_outliers]
        RO_indices = list(DO) + top_k_outliers

        # Step 4: train a classifier on RN + RO with score-derived sample weights
        # Label: cluster id for normals, -1 for outliers
        train_indices = []
        train_labels = []
        train_weights = []

        for i in RN_indices:
            train_indices.append(i)
            train_labels.append(cluster_assignments[i])
            # Weight = rScore: a normal point we are confident about gets high weight
            train_weights.append(max(rscore[i], 1e-6))

        for i in RO_indices:
            train_indices.append(i)
            train_labels.append(-1)
            train_weights.append(max(tscore[i], 1e-6))

        train_indices = np.array(train_indices)
        train_labels = np.array(train_labels)
        train_weights = np.array(train_weights)

        # Edge case: if we somehow only have one class in the training data,
        # the classifier will refuse to train. Just propagate cluster assignments.
        unique_classes = np.unique(train_labels)
        if len(unique_classes) < 2:
            return self._build_result_from_assignments(
                n, cluster_assignments, set(RO_indices),
                rscore, lscore, simscore, tscore,
            )

        X_train = X[train_indices]
        self.classifier.fit(X_train, train_labels, sample_weight=train_weights)

        # Step 5: predict for every point in the dataset
        all_predictions = self.classifier.predict(X)

        # Build the final cluster labels and outlier flags. We OVERRIDE the
        # classifier output with explicit user labels — those are ground truth.
        cluster_labels = np.zeros(n, dtype=int)
        is_outlier = np.zeros(n, dtype=bool)
        for i in range(n):
            if i in DO:
                cluster_labels[i] = -1
                is_outlier[i] = True
            elif i in DN:
                cluster_labels[i] = DN[i]
            elif all_predictions[i] == -1:
                cluster_labels[i] = -1
                is_outlier[i] = True
            else:
                cluster_labels[i] = int(all_predictions[i])

        return SSDBCODIResult(
            cluster_labels=cluster_labels,
            is_outlier=is_outlier,
            rscore=rscore,
            lscore=lscore,
            simscore=simscore,
            tscore=tscore,
        )

    def _build_result_from_assignments(
        self, n, cluster_assignments, outlier_set,
        rscore, lscore, simscore, tscore,
    ) -> SSDBCODIResult:
        """Used when classifier training is impossible (single-class case)."""
        cluster_labels = np.full(n, -1, dtype=int)
        is_outlier = np.zeros(n, dtype=bool)
        for i, c in cluster_assignments.items():
            cluster_labels[i] = c
        for i in outlier_set:
            cluster_labels[i] = -1
            is_outlier[i] = True
        return SSDBCODIResult(
            cluster_labels=cluster_labels,
            is_outlier=is_outlier,
            rscore=rscore,
            lscore=lscore,
            simscore=simscore,
            tscore=tscore,
        )

    def _cold_start_fallback(self, X: np.ndarray, DO: Set[int]) -> SSDBCODIResult:
        """When we have no DN labels yet (initial state), put everything into a
        single cluster and mark only the user-flagged outliers.

        This lets the dashboard render a default scatterplot before the user has
        provided any guidance.
        """
        n = len(X)
        cluster_labels = np.zeros(n, dtype=int)
        is_outlier = np.zeros(n, dtype=bool)
        for i in DO:
            cluster_labels[i] = -1
            is_outlier[i] = True
        zeros = np.zeros(n)
        ones = np.ones(n)
        return SSDBCODIResult(
            cluster_labels=cluster_labels,
            is_outlier=is_outlier,
            rscore=ones,
            lscore=ones,
            simscore=zeros,
            tscore=zeros,
        )
