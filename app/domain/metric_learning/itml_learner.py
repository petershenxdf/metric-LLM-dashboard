"""ITML wrapper.

We use the metric-learn library's ITML implementation under the hood. To
support incremental updates (the dashboard adds constraints one at a time),
we store the cumulative pair list and re-fit ITML each time, passing the
current M as the prior. This mimics online behavior with negligible overhead
for the dataset sizes the dashboard targets.
"""
import warnings
import numpy as np
from typing import List, Tuple

from metric_learn import ITML

from .base import MetricLearner
from app.infrastructure.debug.logger import get_logger


logger = get_logger("metric_learning.itml")


class ITMLLearner(MetricLearner):
    def __init__(self, n_features: int, gamma: float = 1.0, max_iter: int = 500):
        self.gamma = gamma
        self.max_iter = max_iter
        self.M: np.ndarray = np.eye(n_features)
        # Cumulative pair history: list of (i, j) index tuples and ±1 labels
        self.pairs: List[Tuple[int, int]] = []
        self.labels: List[int] = []

    def reset(self, n_features: int) -> None:
        self.M = np.eye(n_features)
        self.pairs = []
        self.labels = []

    def update(self, X: np.ndarray, pairs=None, labels=None, **kwargs) -> None:
        """Add new pair constraints and re-fit ITML using current M as prior.

        Args:
            X: full data matrix.
            pairs: list of (i, j) tuples — point index pairs.
            labels: list of ±1 — 1 for similar, -1 for dissimilar.
        """
        if not pairs or not labels:
            return
        if len(pairs) != len(labels):
            raise ValueError("pairs and labels must have the same length")

        self.pairs.extend(pairs)
        self.labels.extend(labels)

        # Need at least one similar and one dissimilar pair for ITML to work
        unique_labels = set(self.labels)
        if len(unique_labels) < 2:
            # Not enough variety yet — just keep the existing M
            return

        # Build the array form ITML expects: shape (n_pairs, 2, n_features)
        pair_data = np.array([
            [X[i], X[j]] for (i, j) in self.pairs
        ])
        label_arr = np.array(self.labels)

        # Use current M as prior so each fit refines rather than restarts
        itml = ITML(
            gamma=self.gamma,
            max_iter=self.max_iter,
            prior=self.M,
            random_state=42,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                itml.fit(pair_data, label_arr)
                self.M = itml.get_mahalanobis_matrix()
            except Exception as e:
                # ITML can occasionally fail to converge on degenerate inputs.
                # Keep the previous M rather than crashing the pipeline.
                logger.warning("ITML fit failed, keeping previous M: %s", e)

    def get_M(self) -> np.ndarray:
        return self.M.copy()
