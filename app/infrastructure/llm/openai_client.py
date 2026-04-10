"""OpenAI-compatible client.

Works with the official OpenAI API and any provider that exposes the same
/v1/chat/completions interface (DeepSeek, Together, OpenRouter, etc.). The
only difference from the Ollama client is that we send an Authorization
header with the API key.
"""
import requests
from typing import List, Dict

from .base import LLMClient


class OpenAIClient(LLMClient):
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        temperature: float = 0.1,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False,
        }
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("response_format") == "json_object":
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected OpenAI response: {data}") from e

    def is_available(self) -> bool:
        """Light health check using the /models endpoint."""
        if not self.api_key:
            return False
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=3,
            )
            return resp.status_code == 200
        except Exception:
            return False
