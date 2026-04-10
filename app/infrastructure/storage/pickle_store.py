"""Pickle-backed session store -- one file per session.

Survives process restarts. Slightly slower than in-memory. Adequate for
single-machine deployments where you want sessions to persist across reboots.
"""
import pickle
from pathlib import Path
from typing import Optional, List

from .base import SessionStore
from app.models.session_state import SessionState


class PickleSessionStore(SessionStore):
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.pkl"

    def get(self, session_id: str) -> Optional[SessionState]:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, session_id: str, state: SessionState) -> None:
        path = self._path(session_id)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        path.unlink(missing_ok=True)

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()

    def list_sessions(self) -> List[str]:
        return [p.stem for p in self.root.glob("*.pkl")]
