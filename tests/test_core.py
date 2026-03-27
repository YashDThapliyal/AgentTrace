"""Tests for AgentTrace core and the module-level public API."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from agenttrace.config import AgentTraceConfig
from agenttrace.core import AgentTrace
from agenttrace.storage.jsonl import JsonlBackend


def _make_config(tmp_path) -> AgentTraceConfig:
    return AgentTraceConfig(
        backend="jsonl",
        store_path=str(tmp_path / "traces.jsonl"),
        embeddings_provider="local",
        top_k=3,
        threshold=0.0,  # accept all similarities in tests
    )


def _mock_embedder(dim: int = 4) -> MagicMock:
    """Return an embedder that always returns a fixed vector."""
    embedder = MagicMock()
    embedder.embed.return_value = [1.0] + [0.0] * (dim - 1)
    return embedder


# ---------------------------------------------------------------------------
# AgentTrace class
# ---------------------------------------------------------------------------

class TestAgentTraceSave:
    def test_save_returns_valid_uuid(self, tmp_path):
        cfg = _make_config(tmp_path)
        at = AgentTrace(cfg)
        at._embedder = _mock_embedder()
        trace_id = at.save(task="fix bug", reasoning="found it", outcome="fixed")
        assert uuid.UUID(trace_id)  # no exception → valid UUID

    def test_save_persists_trace(self, tmp_path):
        cfg = _make_config(tmp_path)
        at = AgentTrace(cfg)
        at._embedder = _mock_embedder()
        trace_id = at.save(task="fix bug", reasoning="found it", outcome="fixed")
        store = JsonlBackend(cfg.store_path)
        trace = store.get(trace_id)
        assert trace.task == "fix bug"
        assert trace.reasoning == "found it"
        assert trace.outcome == "fixed"

    def test_save_stores_all_optional_fields(self, tmp_path):
        cfg = _make_config(tmp_path)
        at = AgentTrace(cfg)
        at._embedder = _mock_embedder()
        trace_id = at.save(
            task="task",
            reasoning="reason",
            outcome="done",
            errors=["err1"],
            success=False,
            model="gpt-4",
            tags=["python"],
        )
        trace = JsonlBackend(cfg.store_path).get(trace_id)
        assert trace.errors == ["err1"]
        assert trace.success is False
        assert trace.model == "gpt-4"
        assert trace.tags == ["python"]

    def test_save_defaults_empty_errors_and_tags(self, tmp_path):
        cfg = _make_config(tmp_path)
        at = AgentTrace(cfg)
        at._embedder = _mock_embedder()
        trace_id = at.save(task="t", reasoning="r", outcome="o")
        trace = JsonlBackend(cfg.store_path).get(trace_id)
        assert trace.errors == []
        assert trace.tags == []

    def test_save_stores_embedding(self, tmp_path):
        cfg = _make_config(tmp_path)
        at = AgentTrace(cfg)
        at._embedder = _mock_embedder(dim=4)
        trace_id = at.save(task="t", reasoning="r", outcome="o")
        trace = JsonlBackend(cfg.store_path).get(trace_id)
        assert len(trace.embedding) == 4


class TestAgentTraceRecall:
    def test_recall_empty_store_returns_empty_string(self, tmp_path):
        cfg = _make_config(tmp_path)
        at = AgentTrace(cfg)
        at._embedder = _mock_embedder()
        result = at.recall("some task")
        assert result == ""

    def test_recall_returns_formatted_context(self, tmp_path):
        cfg = _make_config(tmp_path)
        at = AgentTrace(cfg)
        at._embedder = _mock_embedder()
        at.save(task="fix thread bug", reasoning="use multiprocessing", outcome="4x speedup")
        result = at.recall("threading performance issue")
        assert "<agent_trace_context>" in result
        assert "fix thread bug" in result

    def test_recall_respects_top_k_override(self, tmp_path):
        cfg = _make_config(tmp_path)
        at = AgentTrace(cfg)
        at._embedder = _mock_embedder()
        for i in range(5):
            at.save(task=f"task {i}", reasoning="r", outcome="o")
        result = at.recall("task", top_k=2)
        assert result.count("[") == 2  # [1] and [2]

    def test_recall_respects_threshold_override(self, tmp_path):
        cfg = _make_config(tmp_path)
        at = AgentTrace(cfg)
        at._embedder = _mock_embedder()
        at.save(task="t", reasoning="r", outcome="o")
        # All stored embeddings match query exactly (similarity=1.0),
        # but threshold=2.0 is impossible → no results
        result = at.recall("t", threshold=2.0)
        assert result == ""

    def test_recall_uses_config_defaults(self, tmp_path):
        cfg = AgentTraceConfig(
            backend="jsonl",
            store_path=str(tmp_path / "traces.jsonl"),
            top_k=1,
            threshold=0.0,
        )
        at = AgentTrace(cfg)
        at._embedder = _mock_embedder()
        for i in range(3):
            at.save(task=f"task {i}", reasoning="r", outcome="o")
        result = at.recall("task")
        assert "[1]" in result
        assert "[2]" not in result


# ---------------------------------------------------------------------------
# Module-level public API (agenttrace.__init__)
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def setup_method(self):
        """Reset the singleton before each test."""
        import agenttrace
        agenttrace._instance = None

    def test_functions_importable(self):
        import agenttrace
        assert callable(agenttrace.init)
        assert callable(agenttrace.recall)
        assert callable(agenttrace.save)

    def test_auto_init_on_recall(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no.json")
        import agenttrace
        agenttrace._instance = None
        # Should not raise — auto-inits with defaults
        result = agenttrace.recall("test task")
        assert result == ""  # empty store

    def test_init_with_overrides(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import agenttrace
        store_path = str(tmp_path / "custom.jsonl")
        agenttrace.init(store_path=store_path, backend="jsonl")
        assert agenttrace._instance is not None
        assert agenttrace._instance._config.store_path == store_path

    def test_init_twice_resets_instance(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import agenttrace
        agenttrace.init(store_path=str(tmp_path / "a.jsonl"))
        first = agenttrace._instance
        agenttrace.init(store_path=str(tmp_path / "b.jsonl"))
        assert agenttrace._instance is not first

    def test_save_and_recall_via_public_api(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import agenttrace
        agenttrace.init(
            store_path=str(tmp_path / "traces.jsonl"),
            backend="jsonl",
            threshold=0.0,
        )
        # Inject a mock embedder so tests don't download sentence-transformers
        agenttrace._instance._embedder = _mock_embedder()  # type: ignore[union-attr]
        trace_id = agenttrace.save(
            task="threading bug",
            reasoning="use multiprocessing",
            outcome="fixed",
        )
        assert uuid.UUID(trace_id)
        result = agenttrace.recall("threading performance")
        assert "<agent_trace_context>" in result
