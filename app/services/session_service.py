"""Session service -- thin wrapper over the SessionStore that adds creation,
reset, and rollback semantics.
"""
import uuid
from typing import Optional

import pandas as pd

from app.infrastructure.storage.base import SessionStore
from app.models.session_state import SessionState


class SessionService:
    def __init__(self, store: SessionStore):
        self.store = store

    def create_session(
        self,
        dataset: pd.DataFrame,
        source_filename: str = "",
        raw_dataset: pd.DataFrame = None,
    ) -> str:
        """Create a new session for a freshly loaded dataset.

        Args:
            dataset: the normalized data matrix the clustering pipeline will consume.
            source_filename: the original file name (for display).
            raw_dataset: the pre-normalized data, preserved for CSV export.
        """
        session_id = uuid.uuid4().hex
        state = SessionState(
            session_id=session_id,
            dataset=dataset,
            source_filename=source_filename,
            raw_dataset=raw_dataset,
        )
        self.store.set(session_id, state)
        return session_id

    def get(self, session_id: str) -> Optional[SessionState]:
        return self.store.get(session_id)

    def save(self, state: SessionState) -> None:
        state.touch()
        self.store.set(state.session_id, state)

    def reset(self, session_id: str) -> None:
        """Wipe all learned state but keep the dataset."""
        state = self.store.get(session_id)
        if state is None:
            return
        state.M = None
        state.DN = {}
        state.DO = set()
        state.constraints_history = []
        state.pending_constraints = []
        state.current_clusters = None
        state.current_outliers = None
        state.current_projection = None
        state.current_scores = None
        state.chat_history = []
        state._snapshots = []
        self.save(state)

    def delete(self, session_id: str) -> None:
        self.store.delete(session_id)

    def rollback(self, session_id: str) -> bool:
        """Undo the last constraint by restoring the previous snapshot."""
        state = self.store.get(session_id)
        if state is None:
            return False
        ok = state.rollback()
        if ok:
            self.save(state)
        return ok
