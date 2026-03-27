"""AgentTrace — reasoning memory for AI agents.

Public API:
    init(**kwargs)   — configure and initialize the singleton instance
    recall(task)     — retrieve similar past traces as a context string
    save(task, ...)  — persist a successful trace
"""
from __future__ import annotations

from agenttrace.config import load_config
from agenttrace.core import AgentTrace

_instance: AgentTrace | None = None


def init(**kwargs: object) -> None:
    """Initialize (or re-initialize) AgentTrace with optional config overrides.

    Keyword arguments are passed directly to load_config() and override all
    other config sources (env vars, config files, defaults).
    """
    global _instance
    _instance = AgentTrace(load_config(**kwargs))


def _get_instance() -> AgentTrace:
    global _instance
    if _instance is None:
        _instance = AgentTrace(load_config())
    return _instance


def recall(
    task: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> str:
    """Return a formatted context block of similar past traces, or '' if none found.

    Auto-initializes with default config if init() has not been called.
    """
    return _get_instance().recall(task, top_k=top_k, threshold=threshold)


def save(
    task: str,
    reasoning: str,
    outcome: str,
    errors: list[str] | None = None,
    success: bool = True,
    model: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Embed and persist a trace. Returns the new trace ID.

    Auto-initializes with default config if init() has not been called.
    """
    return _get_instance().save(
        task=task,
        reasoning=reasoning,
        outcome=outcome,
        errors=errors,
        success=success,
        model=model,
        tags=tags,
    )
