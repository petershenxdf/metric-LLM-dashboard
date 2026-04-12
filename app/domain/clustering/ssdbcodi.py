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
from sklearn.cluster import DBSCAN

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

        # If we have no labels at all, fall back to unsupervised DBSCAN using
        # the learned distance so the dashboard can still show something.
        if not DN:
            return self._unsupervised_fallback(X, DO, distance_func)

        # If the user has only given a single cluster label (e.g. one must-link
        # group), SSDBSCAN expansion would never hit a "different label"
        # terminator and would collapse every reachable point into that one
        # cluster. That destroys the original structure and ignores the learned
        # metric. Fall back to DBSCAN on the learned distance and overlay the
        # user labels on top: the labeled points become a dedicated cluster,
        # everything else is clustered unsupervised.
        if len(set(DN.values())) < 2:
            return self._supervised_overlay_fallback(X, DN, DO, distance_func)

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

    def _run_dbscan_on_distance(
        self, X: np.ndarray, distance_func
    ) -> np.ndarray:
        """Run DBSCAN using the provided distance function (which respects the
        learned Mahalanobis metric). Returns DBSCAN's raw label array.

        eps is auto-estimated from the distribution of k-nearest-neighbor
        distances under the same metric — a classic heuristic that works on
        most datasets without the user having to tune it.
        """
        n = len(X)
        if n < max(self.min_pts, 2):
            return np.full(n, -1, dtype=int)

        dist_matrix = distance_func.pairwise(X)
        # Symmetrize + zero diagonal for DBSCAN's precomputed mode
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
        np.fill_diagonal(dist_matrix, 0.0)

        k = max(self.min_pts, 2)
        sorted_rows = np.sort(dist_matrix, axis=1)
        kth_distances = sorted_rows[:, min(k, n - 1)]
        # 90th percentile of the k-th NN distance: tight enough to separate
        # blobs, loose enough to include core points.
        eps = float(np.percentile(kth_distances, 90))
        if eps <= 0:
            eps = float(np.mean(kth_distances) + 1e-6)

        db = DBSCAN(eps=eps, min_samples=self.min_pts, metric="precomputed").fit(
            dist_matrix
        )
        return db.labels_

    def _unsupervised_fallback(
        self, X: np.ndarray, DO: Set[int], distance_func
    ) -> SSDBCODIResult:
        """Cold-start path: no DN labels. Run DBSCAN on the learned distance
        so the user sees a meaningful initial clustering.
        """
        n = len(X)
        cluster_labels = np.full(n, -1, dtype=int)
        is_outlier = np.zeros(n, dtype=bool)

        raw_labels = self._run_dbscan_on_distance(X, distance_func)

        # Remap DBSCAN labels so cluster ids start at 0 (DBSCAN uses -1 for noise)
        unique_clusters = sorted(set(int(l) for l in raw_labels if l >= 0))
        remap = {old: new for new, old in enumerate(unique_clusters)}
        for i, lbl in enumerate(raw_labels):
            lbl_int = int(lbl)
            if lbl_int < 0:
                cluster_labels[i] = -1
                is_outlier[i] = True
            else:
                cluster_labels[i] = remap[lbl_int]

        # User-flagged outliers always win
        for i in DO:
            if 0 <= i < n:
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

    def _supervised_overlay_fallback(
        self,
        X: np.ndarray,
        DN: Dict[int, int],
        DO: Set[int],
        distance_func,
    ) -> SSDBCODIResult:
        """Handle the "only one unique label in DN" case.

        Rather than letting SSDBSCAN collapse everything into the single
        labeled cluster, we:
          1. Run DBSCAN on the learned distance for structural discovery.
          2. Reserve a dedicated cluster id for the labeled group (the id the
             user chose, offset from DBSCAN's ids if it would collide).
          3. Remap DBSCAN labels to avoid colliding with the labeled id.
          4. Force all DN points into the labeled cluster, force DO into
             outliers, leave everything else as DBSCAN assigned.
        """
        n = len(X)
        cluster_labels = np.full(n, -1, dtype=int)
        is_outlier = np.zeros(n, dtype=bool)

        raw_labels = self._run_dbscan_on_distance(X, distance_func)
        dbscan_ids = sorted(set(int(l) for l in raw_labels if l >= 0))

        # The user-chosen label (there is exactly one unique value in DN)
        user_label = int(next(iter(set(DN.values()))))

        # Build a remap for DBSCAN labels so they skip `user_label`. Cluster
        # ids start at 0 and step over `user_label`.
        remap: Dict[int, int] = {}
        next_id = 0
        for old in dbscan_ids:
            if next_id == user_label:
                next_id += 1
            remap[old] = next_id
            next_id += 1

        for i, lbl in enumerate(raw_labels):
            lbl_int = int(lbl)
            if lbl_int < 0:
                cluster_labels[i] = -1
                is_outlier[i] = True
            else:
                cluster_labels[i] = remap[lbl_int]

        # Overlay DN — user labels take precedence and unify the group
        for i, cid in DN.items():
            if 0 <= i < n:
                cluster_labels[i] = int(cid)
                is_outlier[i] = False

        # DO always wins
        for i in DO:
            if 0 <= i < n:
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
