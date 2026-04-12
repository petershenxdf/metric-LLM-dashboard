"""Pipeline service -- orchestrates one full iteration of the system.

This is the most important file in the backend. It knows the order:
    constraint -> metric learner update (M) -> SSDBCODI (clusters + outliers)
    -> MDS projection -> store results back on the session state

It does NOT know how any of those steps work internally -- it just calls the
domain-layer objects. This keeps the service layer thin and testable.
"""
import time
import numpy as np
from typing import Dict, Any, Optional

from config.config import Config
from app.services.session_service import SessionService
from app.models.session_state import SessionState
from app.domain.clustering.ssdbcodi import SSDBCODI
from app.domain.clustering.distance import MahalanobisDistance, make_distance
from app.domain.metric_learning.composite import CompositeMetricLearner
from app.domain.projection.mds_projector import MDSProjector
from app.domain.constraints.schemas import Constraint
from app.infrastructure.debug.logger import get_logger
from app.infrastructure.debug.debug_recorder import DebugRecorder


logger = get_logger("pipeline")


class PipelineService:
    def __init__(
        self,
        session_service: SessionService,
        config: Config,
        debug_recorder: Optional[DebugRecorder] = None,
    ):
        self.session_service = session_service
        self.config = config
        self.projector = MDSProjector(n_components=2)
        self.debug_recorder = debug_recorder
        # Wired up after construction (to avoid a circular import with
        # FeedbackService, which also depends on PipelineService). app.py
        # sets this attribute once both services exist.
        self.feedback_service = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        session_id: str,
        triggering_constraint: Optional[Constraint] = None,
    ) -> Dict[str, Any]:
        """Run SSDBCODI + MDS with the current M and current DN/DO.

        This is called on the first clustering run and after every constraint
        application (via apply_constraint_and_recluster). The metric learner
        itself is updated separately in the feedback service; this method only
        consumes the current M.

        The optional `triggering_constraint` is passed through to the debug
        recorder so each dumped iteration knows what caused it.
        """
        start = time.time()

        # Flush any pending constraints BEFORE reading state, so the run sees
        # an up-to-date DN / DO / M. This is what makes pressing "Run
        # clustering" the only thing that actually mutates the world.
        n_flushed = 0
        if self.feedback_service is not None:
            n_flushed = self.feedback_service.flush_pending(session_id)

        state = self.session_service.get(session_id)
        if state is None:
            return {"error": f"Session {session_id} not found"}

        # If the caller didn't pass a triggering constraint explicitly but we
        # just flushed some pending ones, attribute the run to the last
        # flushed constraint so the debug recorder has context.
        if triggering_constraint is None and n_flushed > 0 and state.constraints_history:
            triggering_constraint = state.constraints_history[-1]

        X = state.get_X()
        n_features = X.shape[1]

        # Build the distance function from the current M (identity on cold start)
        if state.M is None:
            state.M = np.eye(n_features)
        distance_func = MahalanobisDistance(state.M)

        # Run SSDBCODI
        algo = SSDBCODI(
            min_pts=self.config.ssdbcodi_min_pts,
            alpha=self.config.ssdbcodi_alpha,
            beta=self.config.ssdbcodi_beta,
            k_outliers=self.config.ssdbcodi_k_outliers,
        )
        result = algo.fit(X, state.DN, state.DO, distance_func=distance_func)

        # Run MDS projection
        projection = self.projector.project(X, distance_func)

        # Persist everything on the session state
        state.current_clusters = result.cluster_labels
        state.current_outliers = result.is_outlier
        state.current_projection = projection
        state.current_scores = {
            "rscore": result.rscore,
            "lscore": result.lscore,
            "simscore": result.simscore,
            "tscore": result.tscore,
        }
        self.session_service.save(state)

        duration_ms = (time.time() - start) * 1000.0

        # Compute summary stats for logging
        valid_clusters = result.cluster_labels[result.cluster_labels >= 0]
        n_clusters = int(max(valid_clusters) + 1) if len(valid_clusters) > 0 else 0
        n_outliers = int(sum(result.is_outlier))

        logger.info(
            "Pipeline run: session=%s, clusters=%d, outliers=%d, duration=%.1fms",
            session_id[:8], n_clusters, n_outliers, duration_ms,
        )

        # Dump the iteration for offline inspection if debugging is enabled
        if self.debug_recorder is not None:
            try:
                self.debug_recorder.dump_iteration(
                    state=state,
                    triggering_constraint=triggering_constraint,
                    duration_ms=duration_ms,
                )
            except Exception as e:
                logger.warning("Debug dump failed: %s", e)

        return self._build_response(state)

    def initialize_session(self, session_id: str) -> Dict[str, Any]:
        """First-time run for a newly uploaded dataset.

        Sets M to identity and runs the default pipeline. The user will see
        either a single cluster (if they provided no initial labels) or the
        initial SSDBCODI clustering based on whatever labels were inferred
        from the data file.
        """
        state = self.session_service.get(session_id)
        if state is None:
            return {"error": f"Session {session_id} not found"}

        n_features = state.n_features()
        state.M = np.eye(n_features)
        self.session_service.save(state)

        return self.run_full_pipeline(session_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_response(self, state: SessionState) -> Dict[str, Any]:
        """Build a JSON-serializable response for the frontend."""
        if state.current_projection is None:
            return {"ready": False}

        points = []
        for i in range(len(state.current_projection)):
            points.append({
                "id": int(i),
                "x": float(state.current_projection[i, 0]),
                "y": float(state.current_projection[i, 1]),
                "cluster": int(state.current_clusters[i]),
                "is_outlier": bool(state.current_outliers[i]),
            })

        n_clusters = 0
        if state.current_clusters is not None and len(state.current_clusters) > 0:
            valid_clusters = state.current_clusters[state.current_clusters >= 0]
            if len(valid_clusters) > 0:
                n_clusters = int(max(valid_clusters) + 1)

        return {
            "ready": True,
            "points": points,
            "n_clusters": n_clusters,
            "n_outliers": int(sum(state.current_outliers)),
            "n_constraints": len(state.constraints_history),
            "n_pending": len(state.pending_constraints),
        }

    def build_cluster_summary(self, state: SessionState) -> Dict[str, Any]:
        """Build a compact cluster summary used as context for the chatbox LLM."""
        if state.current_clusters is None:
            return {"status": "not_clustered"}

        cluster_sizes = {}
        for c in state.current_clusters:
            ci = int(c)
            cluster_sizes[ci] = cluster_sizes.get(ci, 0) + 1

        return {
            "n_clusters": len([k for k in cluster_sizes.keys() if k >= 0]),
            "cluster_sizes": cluster_sizes,
            "n_outliers": int(sum(state.current_outliers)),
        }
