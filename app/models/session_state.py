"""SessionState — the bag of state for one analysis session."""
import copy
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any

import numpy as np
import pandas as pd


@dataclass
class SessionState:
    session_id: str
    dataset: pd.DataFrame
    source_filename: str = ""
    # The raw (pre-normalized) data as originally loaded, used for CSV export
    # so users get back their real column values. Optional because older
    # pickled sessions may not have it.
    raw_dataset: Optional[pd.DataFrame] = None

    # Mahalanobis matrix learned over time
    M: Optional[np.ndarray] = None

    # Labels: point_id -> cluster_id (only for points the user explicitly labeled)
    DN: Dict[int, int] = field(default_factory=dict)
    # Set of point_ids the user marked as outliers
    DO: Set[int] = field(default_factory=set)

    # Constraint history (list of Constraint objects, with to_dict()).
    # These are constraints that have already been applied (labels / metric
    # updated and pipeline re-run).
    constraints_history: List[Any] = field(default_factory=list)

    # Pending constraints: staged by the chatbox but not yet applied. The
    # "Run clustering" button is what triggers the flush. This separation
    # lets the user build up several instructions before paying the cost of
    # a pipeline run, and keeps the UX predictable.
    pending_constraints: List[Any] = field(default_factory=list)

    # Most recent clustering output
    current_clusters: Optional[np.ndarray] = None    # shape (n,) cluster_id per point
    current_outliers: Optional[np.ndarray] = None    # shape (n,) bool
    current_projection: Optional[np.ndarray] = None  # shape (n, 2)
    current_scores: Optional[Dict[str, np.ndarray]] = None  # rScore, lScore, simScore, tScore

    # Chat
    chat_history: List[Dict[str, str]] = field(default_factory=list)

    # Snapshots for undo (cap at 10)
    _snapshots: List[Dict[str, Any]] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self):
        self.updated_at = time.time()

    def get_X(self) -> np.ndarray:
        """Return the data matrix as numpy (numeric columns only)."""
        return self.dataset.select_dtypes(include=[np.number]).to_numpy()

    def n_features(self) -> int:
        return self.get_X().shape[1]

    def n_points(self) -> int:
        return len(self.dataset)

    def snapshot(self):
        """Save current mutable state for undo."""
        snap = {
            "M": copy.deepcopy(self.M),
            "DN": copy.deepcopy(self.DN),
            "DO": copy.deepcopy(self.DO),
            "constraints_history": copy.deepcopy(self.constraints_history),
            "pending_constraints": copy.deepcopy(self.pending_constraints),
            "current_clusters": copy.deepcopy(self.current_clusters),
            "current_outliers": copy.deepcopy(self.current_outliers),
            "current_projection": copy.deepcopy(self.current_projection),
            "current_scores": copy.deepcopy(self.current_scores),
        }
        self._snapshots.append(snap)
        if len(self._snapshots) > 10:
            self._snapshots.pop(0)

    def rollback(self) -> bool:
        if not self._snapshots:
            return False
        snap = self._snapshots.pop()
        self.M = snap["M"]
        self.DN = snap["DN"]
        self.DO = snap["DO"]
        self.constraints_history = snap["constraints_history"]
        self.pending_constraints = snap.get("pending_constraints", [])
        self.current_clusters = snap["current_clusters"]
        self.current_outliers = snap["current_outliers"]
        self.current_projection = snap["current_projection"]
        self.current_scores = snap["current_scores"]
        self.touch()
        return True

    def to_summary_dict(self) -> Dict[str, Any]:
        """Lightweight dict for sending to the frontend (no big arrays)."""
        return {
            "session_id": self.session_id,
            "source_filename": self.source_filename,
            "n_points": self.n_points(),
            "n_features": self.n_features(),
            "n_constraints": len(self.constraints_history),
            "n_pending": len(self.pending_constraints),
            "n_labeled_normal": len(self.DN),
            "n_labeled_outlier": len(self.DO),
            "has_clustering": self.current_clusters is not None,
            "has_projection": self.current_projection is not None,
            "n_clusters": int(max(self.current_clusters) + 1) if self.current_clusters is not None and len(self.current_clusters) > 0 else 0,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
