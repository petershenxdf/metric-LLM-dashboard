"""Ollama client.

Talks to a local Ollama server (default http://localhost:11434) using the
OpenAI-compatible /v1/chat/completions endpoint that recent Ollama versions
expose. This way the same response-parsing code works for OpenAI too.
"""
import json
import requests
from typing import List, Dict

from .base import LLMClient


class OllamaClient(LLMClient):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral-small3.1:latest",
        temperature: float = 0.1,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat request and return the assistant's reply text."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False,
        }
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        # OpenAI-compatible response shape
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Ollama response: {data}") from e

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
