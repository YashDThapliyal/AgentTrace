"""Local embedding provider using sentence-transformers."""
from __future__ import annotations

import os
from typing import IO, Any

from agenttrace.embeddings.base import EmbeddingProvider

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalEmbedder(EmbeddingProvider):
    """Generates embeddings locally via sentence-transformers (no API key required).

    The model is lazy-loaded on the first call to embed().
    Default model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        status_io: IO[str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._model: Any = None  # lazy-loaded
        self._status_io = status_io

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            if self._status_io is not None:
                print("Loading embedding model...", file=self._status_io, flush=True)
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        result: list[float] = self._model.encode(text).tolist()
        return result
