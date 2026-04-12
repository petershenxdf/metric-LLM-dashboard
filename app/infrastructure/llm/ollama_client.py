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
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat request and return the assistant's reply text.

        Uses a (connect, read) tuple timeout: the connect phase fails fast
        if Ollama isn't running, but the read phase waits long enough for
        first-call model loads (which can take 60-120s for multi-GB models).
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False,
        }
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=(5, self.timeout),
            )
        except requests.exceptions.ConnectTimeout:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url} (timeout). "
                "Is `ollama serve` running?"
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Is `ollama serve` running?"
            )
        except requests.exceptions.ReadTimeout:
            raise RuntimeError(
                f"Ollama did not respond within {self.timeout}s. The model "
                f"'{self.model}' may be loading into VRAM on the first call, "
                "or it is too large for your GPU and is spilling to CPU RAM. "
                "Try a smaller model or increase LLM_TIMEOUT in .env."
            )

        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama returned 404 for model '{self.model}'. "
                f"Run `ollama pull {self.model}` first, or check LLM_MODEL in .env."
            )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            body = resp.text[:300] if resp.text else "(empty)"
            raise RuntimeError(f"Ollama HTTP {resp.status_code}: {body}") from e

        data = resp.json()
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
