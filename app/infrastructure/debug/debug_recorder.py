"""DebugRecorder -- dumps per-iteration state to disk for offline inspection.

Each dashboard iteration produces a directory like:

    data/debug/<session_id>/iter_0003/
        meta.json        -- iteration metadata (timestamp, duration, constraint, counts)
        M.npy            -- the Mahalanobis matrix at this iteration
        clusters.npy     -- cluster label per point
        outliers.npy     -- outlier flag per point
        projection.npy   -- 2D MDS coordinates per point
        scores.npz       -- rscore, lscore, simscore, tscore
        labels.json      -- the DN and DO sets at this iteration

When DEBUG_DUMP_ENABLED=false in .env, dump_iteration() is a no-op -- the
recorder still exists, so pipeline code does not need conditional logic.

The read methods (list_iterations, load_iteration) work regardless of the
dump flag, because they just read from disk.
"""
import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from app.infrastructure.debug.logger import get_logger


logger = get_logger("debug_recorder")


class DebugRecorder:
    def __init__(self, enabled: bool, dump_dir: str):
        self.enabled = enabled
        self.dump_dir = Path(dump_dir)
        if self.enabled:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Debug dumps enabled at %s", self.dump_dir)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def dump_iteration(
        self,
        state,
        triggering_constraint=None,
        duration_ms: float = 0.0,
    ) -> Optional[Path]:
        """Write a full snapshot of the current session state to disk.

        Args:
            state: the SessionState to dump.
            triggering_constraint: the Constraint that caused this iteration,
                or None for cold-start / plain reruns.
            duration_ms: how long the pipeline run took.

        Returns:
            The path to the new iteration directory, or None if disabled.
        """
        if not self.enabled:
            return None

        iter_num = len(state.constraints_history)
        session_dir = self.dump_dir / state.session_id
        iter_dir = session_dir / f"iter_{iter_num:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # ---- meta.json ---------------------------------------------------
        n_clusters = 0
        if state.current_clusters is not None and len(state.current_clusters) > 0:
            valid = state.current_clusters[state.current_clusters >= 0]
            n_clusters = int(max(valid) + 1) if len(valid) > 0 else 0

        n_outliers = 0
        if state.current_outliers is not None:
            n_outliers = int(sum(state.current_outliers))

        meta = {
            "iteration": iter_num,
            "session_id": state.session_id,
            "timestamp": time.time(),
            "duration_ms": round(duration_ms, 2),
            "triggering_constraint": (
                triggering_constraint.to_dict() if triggering_constraint is not None else None
            ),
            "n_points": state.n_points(),
            "n_features": state.n_features(),
            "n_clusters": n_clusters,
            "n_outliers": n_outliers,
            "n_labeled_normal": len(state.DN),
            "n_labeled_outlier": len(state.DO),
            "source_filename": state.source_filename,
        }
        with open(iter_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # ---- numpy arrays -----------------------------------------------
        if state.M is not None:
            np.save(iter_dir / "M.npy", state.M)
        if state.current_clusters is not None:
            np.save(iter_dir / "clusters.npy", state.current_clusters)
        if state.current_outliers is not None:
            np.save(iter_dir / "outliers.npy", state.current_outliers)
        if state.current_projection is not None:
            np.save(iter_dir / "projection.npy", state.current_projection)
        if state.current_scores:
            np.savez(iter_dir / "scores.npz", **state.current_scores)

        # ---- labels.json -------------------------------------------------
        with open(iter_dir / "labels.json", "w", encoding="utf-8") as f:
            json.dump({
                "DN": {str(k): int(v) for k, v in state.DN.items()},
                "DO": sorted(int(i) for i in state.DO),
            }, f, indent=2)

        logger.debug(
            "Dumped iteration %d for session %s (%.1fms, %d clusters, %d outliers)",
            iter_num, state.session_id, duration_ms, n_clusters, n_outliers,
        )
        return iter_dir

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def list_iterations(self, session_id: str) -> List[Dict[str, Any]]:
        """Return a list of meta.json dicts for all dumped iterations of a session."""
        session_dir = self.dump_dir / session_id
        if not session_dir.exists():
            return []

        iterations = []
        for iter_dir in sorted(session_dir.iterdir()):
            if not iter_dir.is_dir():
                continue
            meta_file = iter_dir / "meta.json"
            if not meta_file.exists():
                continue
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    iterations.append(json.load(f))
            except Exception as e:
                logger.warning("Failed to read %s: %s", meta_file, e)
        return iterations

    def load_iteration(self, session_id: str, iter_num: int) -> Dict[str, Any]:
        """Return a dict containing meta + loaded numpy arrays for one iteration."""
        iter_dir = self.dump_dir / session_id / f"iter_{iter_num:04d}"
        if not iter_dir.exists():
            raise FileNotFoundError(f"No such iteration: {iter_dir}")

        result: Dict[str, Any] = {}

        meta_file = iter_dir / "meta.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                result["meta"] = json.load(f)

        for fname, key in [
            ("M.npy", "M"),
            ("clusters.npy", "clusters"),
            ("outliers.npy", "outliers"),
            ("projection.npy", "projection"),
        ]:
            path = iter_dir / fname
            if path.exists():
                result[key] = np.load(path)

        scores_path = iter_dir / "scores.npz"
        if scores_path.exists():
            with np.load(scores_path) as npz:
                result["scores"] = {k: npz[k] for k in npz.files}

        labels_path = iter_dir / "labels.json"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                result["labels"] = json.load(f)

        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> None:
        """Delete all dumped iterations for a session."""
        session_dir = self.dump_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info("Cleared debug dumps for session %s", session_id)
