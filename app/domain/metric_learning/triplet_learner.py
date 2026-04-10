"""Triplet-based metric learner using simple projected gradient descent.

Each triplet (anchor, positive, negative) gives one constraint:
    d_M(anchor, positive)^2 + margin < d_M(anchor, negative)^2

We minimize a hinge loss on the violation and project M back onto the
positive semi-definite cone after each step. This is a deliberately simple
update — for the dashboard's interactive use case, single-step updates that
respond to one user input at a time are exactly what we want.
"""
import numpy as np

from .base import MetricLearner


class TripletLearner(MetricLearner):
    def __init__(self, n_features: int, lr: float = 0.01, margin: float = 1.0):
        self.lr = lr
        self.margin = margin
        self.M: np.ndarray = np.eye(n_features)

    def reset(self, n_features: int) -> None:
        self.M = np.eye(n_features)

    def update(self, X: np.ndarray, anchor=None, positive=None, negative=None, **kwargs) -> None:
        """Apply one triplet constraint via a single SGD step.

        We use the gradient of the hinge loss
            L = max(0, margin + d_M(a, p)^2 - d_M(a, n)^2)
        which yields
            dL/dM = (a - p)(a - p)^T - (a - n)(a - n)^T
        when the constraint is violated, and zero otherwise.
        """
        if anchor is None or positive is None or negative is None:
            return

        a = X[anchor]
        p = X[positive]
        n_vec = X[negative]

        diff_pos = a - p
        diff_neg = a - n_vec

        d_pos_sq = diff_pos @ self.M @ diff_pos
        d_neg_sq = diff_neg @ self.M @ diff_neg

        # Only update if the margin is violated
        if d_pos_sq + self.margin > d_neg_sq:
            grad = np.outer(diff_pos, diff_pos) - np.outer(diff_neg, diff_neg)
            self.M = self.M - self.lr * grad
            # Project back onto the PSD cone (clip negative eigenvalues)
            self.M = self._project_to_psd(self.M)

    def get_M(self) -> np.ndarray:
        return self.M.copy()

    @staticmethod
    def _project_to_psd(M: np.ndarray) -> np.ndarray:
        """Symmetrize and clip negative eigenvalues to keep M positive semi-definite."""
        M_sym = (M + M.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(M_sym)
        eigvals = np.maximum(eigvals, 1e-8)
        return (eigvecs * eigvals) @ eigvecs.T

    def set_M(self, M_new: np.ndarray) -> None:
        """Allow the composite learner to sync the shared M after another
        sub-learner has updated it."""
        self.M = M_new.copy()
