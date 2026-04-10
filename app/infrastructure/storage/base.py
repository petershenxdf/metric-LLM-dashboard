"""Abstract base class for session storage backends."""
from abc import ABC, abstractmethod
from typing import Optional, List

from app.models.session_state import SessionState


class SessionStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> Optional[SessionState]:
        ...

    @abstractmethod
    def set(self, session_id: str, state: SessionState) -> None:
        ...

    @abstractmethod
    def delete(self, session_id: str) -> None:
        ...

    @abstractmethod
    def exists(self, session_id: str) -> bool:
        ...

    @abstractmethod
    def list_sessions(self) -> List[str]:
        ...
