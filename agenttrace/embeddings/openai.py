"""OpenAI embedding provider."""
from __future__ import annotations

import os

from agenttrace.embeddings.base import EmbeddingProvider

_DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbedder(EmbeddingProvider):
    """Generates embeddings via the OpenAI API.

    Requires: pip install openai
    Requires: OPENAI_API_KEY environment variable (or pass api_key explicitly).
    Default model: text-embedding-3-small (1536 dimensions).
    """

    def __init__(self, model: str = _DEFAULT_MODEL, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key= to OpenAIEmbedder()."
            )

    def embed(self, text: str) -> list[float]:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI embeddings. "
                "Install with: pip install openai"
            ) from exc
        client = openai.OpenAI(api_key=self._api_key)
        response = client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding
