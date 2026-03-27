"""Shared data model and abstract storage interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Trace:
    id: str
    task: str
    reasoning: str
    outcome: str
    timestamp: str  # ISO 8601 UTC, e.g. "2025-01-01T00:00:00Z"
    errors: list[str] = field(default_factory=list)
    success: bool = True
    model: str | None = None
    tags: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)


class StorageBackend(ABC):
    """Abstract interface all storage backends must implement."""

    @abstractmethod
    def save(self, trace: Trace) -> None:
        """Persist a trace."""

    @abstractmethod
    def get(self, id: str) -> Trace:
        """Return a trace by ID. Raises KeyError if not found."""

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a trace by ID. Raises KeyError if not found."""

    @abstractmethod
    def all_embeddings(self) -> list[tuple[str, list[float]]]:
        """Return (id, embedding) pairs for all traces (efficient bulk read)."""

    @abstractmethod
    def list(self) -> list[Trace]:
        """Return all stored traces."""
