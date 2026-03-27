"""Semantic retrieval via cosine similarity.

Pure functions — no side effects, no I/O.
"""
from __future__ import annotations

import numpy as np


def rank(
    query_embedding: list[float],
    stored_embeddings: list[tuple[str, list[float]]],
    top_k: int = 3,
    threshold: float = 0.75,
) -> list[tuple[str, float]]:
    """Return top-k traces above threshold, sorted by cosine similarity descending.

    Args:
        query_embedding: Embedding of the current task description.
        stored_embeddings: List of (id, embedding) pairs from the store.
        top_k: Maximum number of results to return.
        threshold: Minimum cosine similarity score required.

    Returns:
        List of (id, score) pairs, sorted by score descending.
    """
    if not stored_embeddings or top_k == 0:
        return []

    query = np.array(query_embedding, dtype=np.float64)
    query_norm = float(np.linalg.norm(query))

    results: list[tuple[str, float]] = []
    for trace_id, embedding in stored_embeddings:
        vec = np.array(embedding, dtype=np.float64)
        vec_norm = float(np.linalg.norm(vec))
        if query_norm == 0.0 or vec_norm == 0.0:
            score = 0.0
        else:
            score = float(np.dot(query, vec) / (query_norm * vec_norm))
        results.append((trace_id, score))

    results.sort(key=lambda x: x[1], reverse=True)
    filtered = [(id_, s) for id_, s in results if s >= threshold]
    return filtered[:top_k]
