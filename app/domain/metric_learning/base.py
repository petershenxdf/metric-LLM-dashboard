"""Abstract base class for metric learners.

A metric learner maintains a Mahalanobis matrix M that defines a learned
distance d_M(p, q) = sqrt((p-q)^T M (p-q)). Each subclass implements `update`
to consume one constraint at a time and adjust M accordingly.
"""
from abc import ABC, abstractmethod
import numpy as np


class MetricLearner(ABC):
    @abstractmethod
    def update(self, X: np.ndarray, **kwargs) -> None:
        """Apply one constraint and update the internal M matrix.

        Subclasses define their own kwargs for the constraint payload (e.g.
        pairs+labels for ITML, anchor/positive/negative for triplet).
        """
        ...

    @abstractmethod
    def get_M(self) -> np.ndarray:
        """Return a copy of the current M matrix."""
        ...

    @abstractmethod
    def reset(self, n_features: int) -> None:
        """Reset M to the identity matrix (start from Euclidean)."""
        ...
