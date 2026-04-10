"""Feedback service -- apply a structured constraint to the session state.

This is where the two channels (LABEL and METRIC) are wired up. Given a
Constraint, we:

1. Route it to the correct channel(s).
2. For the label channel, directly update state.DN / state.DO / cluster ids.
3. For the metric channel, feed it into the composite metric learner and
   extract the updated M matrix.
4. Call the pipeline service to re-cluster and re-project.
5. Return the fresh result to the frontend.

The composite metric learner is cached per session (lazy-instantiated on
first use) so it accumulates across constraint updates.
"""
from typing import Dict, Any

import numpy as np

from config.config import Config
from app.services.session_service import SessionService
from app.services.pipeline_service import PipelineService
from app.models.session_state import SessionState
from app.domain.constraints.schemas import (
    Constraint,
    MustLink,
    CannotLink,
    Triplet,
    ClusterCount,
    OutlierLabel,
    FeatureHint,
    ClusterMerge,
    Reassign,
)
from app.domain.constraints.router import route_constraint, ChannelType
from app.domain.metric_learning.composite import CompositeMetricLearner


class FeedbackService:
    def __init__(
        self,
        session_service: SessionService,
        pipeline_service: PipelineService,
        config: Config,
    ):
        self.session_service = session_service
        self.pipeline_service = pipeline_service
        self.config = config
        # session_id -> CompositeMetricLearner cache
        self._learners: Dict[str, CompositeMetricLearner] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_constraint(self, session_id: str, constraint: Constraint) -> Dict[str, Any]:
        state = self.session_service.get(session_id)
        if state is None:
            return {"error": f"Session {session_id} not found"}

        # Snapshot for undo BEFORE any mutation
        state.snapshot()

        channel = route_constraint(constraint)

        # Apply to label channel if applicable
        if channel in (ChannelType.LABEL, ChannelType.BOTH):
            self._apply_to_label_channel(state, constraint)

        # Apply to metric channel if applicable
        updated_M = None
        if channel in (ChannelType.METRIC, ChannelType.BOTH):
            updated_M = self._apply_to_metric_channel(state, constraint)

        self.session_service.save(state)

        # Re-run the pipeline with the updated labels and/or metric
        return self.pipeline_service.apply_constraint_and_recluster(
            session_id, constraint, updated_M=updated_M
        )

    def undo_last(self, session_id: str) -> Dict[str, Any]:
        """Undo the last constraint by rolling back the session snapshot,
        then re-running the pipeline from the rolled-back state.
        """
        ok = self.session_service.rollback(session_id)
        if not ok:
            return {"error": "Nothing to undo"}

        # Rebuild the metric learner cache to match the rolled-back M.
        # The simplest correct thing: drop the cached learner so the next
        # constraint rebuilds it from the (rolled-back) state.
        self._learners.pop(session_id, None)

        return self.pipeline_service.run_full_pipeline(session_id)

    # ------------------------------------------------------------------
    # Label channel
    # ------------------------------------------------------------------

    def _apply_to_label_channel(self, state: SessionState, constraint: Constraint) -> None:
        if isinstance(constraint, MustLink):
            self._apply_must_link_labels(state, constraint)
        elif isinstance(constraint, OutlierLabel):
            self._apply_outlier_label(state, constraint)
        elif isinstance(constraint, ClusterMerge):
            self._apply_cluster_merge(state, constraint)
        elif isinstance(constraint, Reassign):
            self._apply_reassign(state, constraint)
        elif isinstance(constraint, ClusterCount):
            # cluster_count does not modify labels directly; the algorithm
            # uses it as a hint during its next run. We store it as state.
            # For now, the simplest implementation: just record it in history,
            # which is already done by the pipeline service.
            pass

    def _apply_must_link_labels(self, state: SessionState, c: MustLink) -> None:
        """Give all points in the must-link set a shared cluster label.

        Logic: if any of the points already has a label in DN, use that as
        the target. Otherwise, pick a new cluster id (max existing + 1).
        Remove the points from DO if they were marked as outliers.
        """
        existing_labels = [state.DN[i] for i in c.point_ids if i in state.DN]
        if existing_labels:
            target_label = existing_labels[0]
        else:
            target_label = self._next_cluster_id(state)

        for i in c.point_ids:
            state.DN[i] = target_label
            state.DO.discard(i)

    def _apply_outlier_label(self, state: SessionState, c: OutlierLabel) -> None:
        if c.is_outlier:
            for i in c.point_ids:
                state.DO.add(i)
                state.DN.pop(i, None)
        else:
            for i in c.point_ids:
                state.DO.discard(i)

    def _apply_cluster_merge(self, state: SessionState, c: ClusterMerge) -> None:
        """Merge multiple clusters into one by rewriting DN labels."""
        if len(c.cluster_ids) < 2:
            return
        target = c.cluster_ids[0]
        to_merge = set(c.cluster_ids[1:])

        # Rewrite labels in DN
        for pid, cid in list(state.DN.items()):
            if cid in to_merge:
                state.DN[pid] = target

        # Also rewrite the existing cluster-label array so the frontend shows
        # the merge immediately even before re-clustering finishes.
        if state.current_clusters is not None:
            for i in range(len(state.current_clusters)):
                if int(state.current_clusters[i]) in to_merge:
                    state.current_clusters[i] = target

    def _apply_reassign(self, state: SessionState, c: Reassign) -> None:
        for i in c.point_ids:
            state.DN[i] = c.target_cluster_id
            state.DO.discard(i)

    def _next_cluster_id(self, state: SessionState) -> int:
        existing = set(state.DN.values())
        if state.current_clusters is not None:
            for cid in state.current_clusters:
                if int(cid) >= 0:
                    existing.add(int(cid))
        return max(existing) + 1 if existing else 0

    # ------------------------------------------------------------------
    # Metric channel
    # ------------------------------------------------------------------

    def _apply_to_metric_channel(self, state: SessionState, constraint: Constraint) -> np.ndarray:
        """Update the composite metric learner with this constraint and
        return the fresh M matrix.
        """
        learner = self._get_or_create_learner(state)
        X = state.get_X()
        learner.update(X, constraint=constraint)
        return learner.get_M()

    def _get_or_create_learner(self, state: SessionState) -> CompositeMetricLearner:
        sid = state.session_id
        if sid not in self._learners:
            n_features = state.n_features()
            feature_names = list(state.dataset.columns)
            learner = CompositeMetricLearner(
                n_features=n_features,
                feature_names=feature_names,
                itml_gamma=self.config.itml_gamma,
                triplet_lr=self.config.triplet_lr,
            )
            # Initialize the learner's M to match whatever the state currently holds
            if state.M is not None:
                learner.M = state.M.copy()
                learner._sync_M()
            self._learners[sid] = learner
        return self._learners[sid]
