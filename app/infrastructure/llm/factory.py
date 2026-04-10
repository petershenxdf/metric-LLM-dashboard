"""LLM factory -- pick a client based on the config.

Adding a new provider only requires:
1. Implementing LLMClient in a new file.
2. Adding one branch here.
No other code changes needed -- this is the dependency-inversion seam.
"""
from .base import LLMClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient


def create_llm_client(config) -> LLMClient:
    """Build an LLMClient from the project config."""
    provider = config.llm_provider.lower()

    if provider == "ollama":
        return OllamaClient(
            base_url=config.llm_base_url,
            model=config.llm_model,
            temperature=config.llm_temperature,
            timeout=config.llm_timeout,
        )

    if provider == "openai":
        return OpenAIClient(
            base_url=config.llm_base_url,
            model=config.llm_model,
            api_key=config.llm_api_key,
            temperature=config.llm_temperature,
            timeout=config.llm_timeout,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
