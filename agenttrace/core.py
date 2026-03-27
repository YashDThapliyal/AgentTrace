"""AgentTrace core — wires config, storage, embeddings, retrieval, and injection."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import IO

from agenttrace.config import AgentTraceConfig
from agenttrace.embeddings.base import EmbeddingProvider, get_provider
from agenttrace.injection import format_traces
from agenttrace.retrieval import rank
from agenttrace.storage.base import StorageBackend, Trace
from agenttrace.storage.jsonl import JsonlBackend
from agenttrace.storage.sqlite import SqliteBackend


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AgentTrace:
    """Orchestrates save and recall flows for the AgentTrace library."""

    def __init__(self, config: AgentTraceConfig, status_io: IO[str] | None = None) -> None:
        self._config = config
        self._storage: StorageBackend | None = None
        self._embedder: EmbeddingProvider | None = None
        self._status_io = status_io

    def _get_storage(self) -> StorageBackend:
        if self._storage is None:
            backend = self._config.backend
            path = self._config.store_path
            if backend == "sqlite":
                self._storage = SqliteBackend(path)
            elif backend == "jsonl":
                self._storage = JsonlBackend(path)
            else:
                raise ValueError(
                    f"Unknown backend: {backend!r}. Valid options: 'jsonl', 'sqlite'."
                )
        return self._storage

    def _get_embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = get_provider(self._config, status_io=self._status_io)
        return self._embedder

    def save(
        self,
        task: str,
        reasoning: str,
        outcome: str,
        errors: list[str] | None = None,
        success: bool = True,
        model: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Embed and persist a trace. Returns the new trace ID."""
        embedding = self._get_embedder().embed(task)
        trace = Trace(
            id=str(uuid.uuid4()),
            task=task,
            reasoning=reasoning,
            outcome=outcome,
            errors=errors or [],
            success=success,
            model=model,
            timestamp=_now_utc(),
            tags=tags or [],
            embedding=embedding,
        )
        self._get_storage().save(trace)
        return trace.id

    def recall(
        self,
        task_description: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> str:
        """Return a formatted context block of similar past traces, or '' if none."""
        effective_top_k = top_k if top_k is not None else self._config.top_k
        effective_threshold = threshold if threshold is not None else self._config.threshold

        query_embedding = self._get_embedder().embed(task_description)
        all_embs = self._get_storage().all_embeddings()
        ranked = rank(
            query_embedding, all_embs, top_k=effective_top_k, threshold=effective_threshold
        )

        traces_with_scores: list[tuple[Trace, float]] = []
        storage = self._get_storage()
        for trace_id, score in ranked:
            try:
                traces_with_scores.append((storage.get(trace_id), score))
            except KeyError:
                continue

        return format_traces(traces_with_scores)
