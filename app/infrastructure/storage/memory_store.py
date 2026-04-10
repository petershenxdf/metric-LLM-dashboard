"""Simple in-memory session store -- a dict.

Good enough for single-process development. Not suitable for multi-worker
deployments because workers don't share state. Switch to PickleSessionStore
or a Redis-backed implementation for production.
"""
from typing import Dict, Optional, List

from .base import SessionStore
from app.models.session_state import SessionState


class InMemorySessionStore(SessionStore):
    def __init__(self):
        self._store: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._store.get(session_id)

    def set(self, session_id: str, state: SessionState) -> None:
        self._store[session_id] = state

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        return session_id in self._store

    def list_sessions(self) -> List[str]:
        return list(self._store.keys())
