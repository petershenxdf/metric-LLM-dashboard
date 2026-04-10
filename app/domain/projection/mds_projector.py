"""MDS projection using a precomputed Mahalanobis distance matrix.

The dashboard scatterplot needs 2D coordinates for every point. Every time M
is updated, we re-project so the visual layout reflects the learned metric.
"""
import warnings

import numpy as np
from sklearn.manifold import MDS


class MDSProjector:
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def project(self, X: np.ndarray, distance_func) -> np.ndarray:
        """Project X to n_components dims using the given distance function.

        Args:
            X: data matrix (n, d).
            distance_func: object with `pairwise(X)` returning an (n, n) array.

        Returns:
            ndarray of shape (n, n_components).
        """
        dist_matrix = distance_func.pairwise(X)

        # Symmetrize and zero out the diagonal — MDS is sensitive to numerical
        # noise that breaks these invariants.
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
        np.fill_diagonal(dist_matrix, 0.0)

        mds = MDS(
            n_components=self.n_components,
            dissimilarity="precomputed",
            random_state=self.random_state,
            normalized_stress="auto",
            n_init=1,           # speed: one initialization is enough for the dashboard
            max_iter=200,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            coords = mds.fit_transform(dist_matrix)
        return coords
