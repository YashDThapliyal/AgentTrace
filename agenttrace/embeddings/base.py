"""Abstract embedding provider interface and factory."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import IO

from agenttrace.config import AgentTraceConfig


class EmbeddingProvider(ABC):
    """Abstract interface all embedding providers must implement."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a fixed-length float vector for the given text."""


def get_provider(
    config: AgentTraceConfig,
    status_io: IO[str] | None = None,
) -> EmbeddingProvider:
    """Instantiate the embedding provider specified in config."""
    provider = config.embeddings_provider
    if provider == "local":
        from agenttrace.embeddings.local import LocalEmbedder
        return LocalEmbedder(status_io=status_io)
    if provider == "openai":
        from agenttrace.embeddings.openai import OpenAIEmbedder
        return OpenAIEmbedder()
    if provider == "anthropic":
        from agenttrace.embeddings.anthropic import AnthropicEmbedder
        return AnthropicEmbedder()
    raise ValueError(
        f"Unknown embeddings provider: {provider!r}. "
        "Valid options: 'local', 'openai', 'anthropic'."
    )
