"""Anthropic embedding provider (via Voyage AI)."""
from __future__ import annotations

import os

from agenttrace.embeddings.base import EmbeddingProvider

_DEFAULT_MODEL = "voyage-3-lite"


class AnthropicEmbedder(EmbeddingProvider):
    """Generates embeddings via Voyage AI (Anthropic-recommended embedding service).

    Requires: pip install voyageai
    Requires: ANTHROPIC_API_KEY or VOYAGE_API_KEY environment variable
              (or pass api_key= explicitly).
    Default model: voyage-3-lite.
    """

    def __init__(self, model: str = _DEFAULT_MODEL, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("VOYAGE_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "An API key is required for the Anthropic/Voyage embedding provider. "
                "Set ANTHROPIC_API_KEY or VOYAGE_API_KEY, or pass api_key= explicitly."
            )

    def embed(self, text: str) -> list[float]:
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "voyageai package is required for Anthropic embeddings. "
                "Install with: pip install voyageai"
            ) from exc
        client = voyageai.Client(api_key=self._api_key)
        result = client.embed([text], model=self._model)
        embedding: list[float] = result.embeddings[0]
        return embedding
