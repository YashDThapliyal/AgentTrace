"""Tests for the CLI using click.testing.CliRunner."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import agenttrace as _api
from agenttrace.cli import _get_core, cli
from agenttrace.config import AgentTraceConfig
from agenttrace.core import AgentTrace


def _make_at(tmp_path) -> AgentTrace:
    cfg = AgentTraceConfig(
        backend="jsonl",
        store_path=str(tmp_path / "traces.jsonl"),
        threshold=0.0,
        top_k=5,
    )
    at = AgentTrace(cfg)
    at._embedder = MagicMock()
    at._embedder.embed.return_value = [1.0, 0.0, 0.0, 0.0]
    return at


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def at_with_trace(tmp_path):
    at = _make_at(tmp_path)
    trace_id = at.save(
        task="fix the threading bug",
        reasoning="switched to multiprocessing",
        outcome="4x speedup",
        tags=["python", "concurrency"],
    )
    return at, trace_id


# ---------------------------------------------------------------------------
# _get_core
# ---------------------------------------------------------------------------

class TestGetCore:
    def test_sets_status_io_to_stderr(self, monkeypatch):
        mock_instance = MagicMock()
        monkeypatch.setattr(_api, "_get_instance", lambda: mock_instance)
        result = _get_core()
        assert result._status_io is sys.stderr


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

class TestRecallCommand:
    def test_recall_with_match(self, runner, at_with_trace, monkeypatch):
        at, _ = at_with_trace
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, ["recall", "threading performance"])
        assert result.exit_code == 0
        assert "<agent_trace_context>" in result.output

    def test_recall_no_match(self, runner, tmp_path, monkeypatch):
        at = _make_at(tmp_path)
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, ["recall", "completely unrelated topic"])
        assert result.exit_code == 0
        assert "No matching traces found" in result.output

    def test_recall_top_k_option(self, runner, at_with_trace, monkeypatch):
        at, _ = at_with_trace
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, ["recall", "bug", "--top-k", "1"])
        assert result.exit_code == 0

    def test_recall_threshold_option(self, runner, tmp_path, monkeypatch):
        at = _make_at(tmp_path)
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, ["recall", "bug", "--threshold", "0.99"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

class TestSaveCommand:
    def test_save_outputs_trace_id(self, runner, tmp_path, monkeypatch):
        at = _make_at(tmp_path)
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, [
            "save",
            "--task", "my task",
            "--reasoning", "my reasoning",
            "--outcome", "my outcome",
        ])
        assert result.exit_code == 0
        assert "Trace saved:" in result.output

    def test_save_with_all_options(self, runner, tmp_path, monkeypatch):
        at = _make_at(tmp_path)
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, [
            "save",
            "--task", "task",
            "--reasoning", "reason",
            "--outcome", "done",
            "--errors", "err1",
            "--errors", "err2",
            "--tags", "python",
            "--tags", "async",
            "--model", "claude",
            "--success",
        ])
        assert result.exit_code == 0
        assert "Trace saved:" in result.output

    def test_save_missing_required_args_fails(self, runner, monkeypatch):
        result = runner.invoke(cli, ["save", "--task", "only task"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

class TestListCommand:
    def test_list_shows_traces(self, runner, at_with_trace, monkeypatch):
        at, _ = at_with_trace
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "fix the threading bug" in result.output

    def test_list_empty_store(self, runner, tmp_path, monkeypatch):
        at = _make_at(tmp_path)
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No traces stored yet" in result.output


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

class TestInspectCommand:
    def test_inspect_known_id(self, runner, at_with_trace, monkeypatch):
        at, trace_id = at_with_trace
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, ["inspect", trace_id])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["id"] == trace_id

    def test_inspect_unknown_id(self, runner, tmp_path, monkeypatch):
        at = _make_at(tmp_path)
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, ["inspect", "nonexistent-id"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

class TestPruneCommand:
    def test_prune_old_traces(self, runner, at_with_trace, monkeypatch):
        at, _ = at_with_trace
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        # days=0 means delete traces older than today — all traces qualify
        result = runner.invoke(cli, ["prune", "--days", "0"])
        assert result.exit_code == 0
        assert "Pruned" in result.output

    def test_prune_nothing_to_prune(self, runner, at_with_trace, monkeypatch):
        at, _ = at_with_trace
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        # days=999 means delete traces older than 999 days — nothing qualifies
        result = runner.invoke(cli, ["prune", "--days", "999"])
        assert result.exit_code == 0
        assert "No traces to prune" in result.output


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStatsCommand:
    def test_stats_shows_count(self, runner, at_with_trace, monkeypatch):
        at, _ = at_with_trace
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "1" in result.output  # at least 1 trace

    def test_stats_empty_store(self, runner, tmp_path, monkeypatch):
        at = _make_at(tmp_path)
        monkeypatch.setattr("agenttrace.cli._get_core", lambda: at)
        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "0" in result.output
