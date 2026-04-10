"""Abstract base class for LLM clients.

Any concrete client (Ollama, OpenAI, Anthropic, ...) must implement two
methods. The dashboard service code only depends on this interface, so
swapping providers is a config-only change.
"""
from abc import ABC, abstractmethod
from typing import List, Dict


class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a list of {role, content} messages and return the assistant's reply.

        Args:
            messages: list of message dicts in OpenAI format.
            **kwargs: optional overrides (temperature, max_tokens, ...).

        Returns:
            The assistant's text response.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Quick health check -- can we reach the model right now?"""
        ...
