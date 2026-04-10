"""Distance functions used by SSDBCODI.

The clustering algorithm only depends on a callable `distance_func(p, q) -> float`
or a method `pairwise(X) -> ndarray`. This lets us swap Euclidean for a learned
Mahalanobis metric without touching the algorithm code.
"""
import numpy as np
from typing import Callable
from sklearn.metrics import pairwise_distances


def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Plain Euclidean distance between two 1-D vectors."""
    diff = p - q
    return float(np.sqrt(np.dot(diff, diff)))


def euclidean_pairwise(X: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix."""
    return pairwise_distances(X, metric="euclidean")


class MahalanobisDistance:
    """Mahalanobis distance parameterized by a positive semi-definite matrix M.

    d_M(p, q) = sqrt((p - q)^T M (p - q))

    Equivalent to Euclidean distance in the linearly transformed space x' = L x,
    where M = L L^T (Cholesky factor). We precompute L for efficient pairwise
    computation via plain Euclidean on the transformed data.
    """

    def __init__(self, M: np.ndarray):
        self.M = None
        self.L = None
        self.update_M(M)

    def update_M(self, M_new: np.ndarray):
        """Replace the metric matrix and refresh the Cholesky factor."""
        # Make sure M is symmetric and positive definite. Add a tiny ridge
        # term in case the new M came from a noisy update step.
        M_sym = (M_new + M_new.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(M_sym)
        eigvals = np.maximum(eigvals, 1e-8)
        self.M = (eigvecs * eigvals) @ eigvecs.T
        # Cholesky for fast pairwise computation
        try:
            self.L = np.linalg.cholesky(self.M)
        except np.linalg.LinAlgError:
            # Fall back: rebuild with stronger ridge
            self.M = self.M + 1e-4 * np.eye(self.M.shape[0])
            self.L = np.linalg.cholesky(self.M)

    def __call__(self, p: np.ndarray, q: np.ndarray) -> float:
        diff = p - q
        return float(np.sqrt(diff @ self.M @ diff))

    def pairwise(self, X: np.ndarray) -> np.ndarray:
        """Compute the full pairwise distance matrix using the Cholesky trick.

        Transform X to X' = X L^T, then plain Euclidean in X' space equals
        Mahalanobis in X space.
        """
        X_transformed = X @ self.L
        return pairwise_distances(X_transformed, metric="euclidean")

    def get_M(self) -> np.ndarray:
        return self.M.copy()


def make_distance(M: np.ndarray = None, n_features: int = None) -> MahalanobisDistance:
    """Convenience factory: build a MahalanobisDistance from M, or identity."""
    if M is None:
        if n_features is None:
            raise ValueError("Either M or n_features must be provided")
        M = np.eye(n_features)
    return MahalanobisDistance(M)
