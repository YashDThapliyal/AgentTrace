"""Tests for injection.format_traces()."""
from agenttrace.injection import format_traces
from agenttrace.storage.base import Trace


def make_trace(**kwargs) -> Trace:
    defaults = dict(
        id="abc123",
        task="fix the threading bug",
        reasoning="identified GIL contention",
        outcome="4x speedup",
        timestamp="2025-01-01T00:00:00Z",
        errors=["tried thread pool, no improvement"],
        tags=["python", "concurrency"],
        embedding=[0.1, 0.2],
    )
    defaults.update(kwargs)
    return Trace(**defaults)


class TestEmptyInput:
    def test_empty_list_returns_empty_string(self):
        assert format_traces([]) == ""


class TestSingleTrace:
    def test_output_wrapped_in_xml_tags(self):
        t = make_trace()
        result = format_traces([(t, 0.91)])
        assert result.startswith("<agent_trace_context>")
        assert result.strip().endswith("</agent_trace_context>")

    def test_similarity_formatted_to_two_decimals(self):
        t = make_trace()
        result = format_traces([(t, 0.91234)])
        assert "similarity: 0.91" in result
        assert "0.91234" not in result

    def test_trace_numbered_one(self):
        t = make_trace()
        result = format_traces([(t, 0.9)])
        assert "[1]" in result

    def test_task_included(self):
        t = make_trace(task="my unique task string")
        result = format_traces([(t, 0.9)])
        assert "my unique task string" in result

    def test_outcome_included_as_resolution(self):
        t = make_trace(outcome="my unique outcome")
        result = format_traces([(t, 0.9)])
        assert "my unique outcome" in result

    def test_tags_shown_when_present(self):
        t = make_trace(tags=["python", "concurrency"])
        result = format_traces([(t, 0.9)])
        assert "python" in result
        assert "concurrency" in result

    def test_tags_omitted_when_empty(self):
        t = make_trace(tags=[])
        result = format_traces([(t, 0.9)])
        assert "tags:" not in result

    def test_errors_shown_when_present(self):
        t = make_trace(errors=["tried X, failed"])
        result = format_traces([(t, 0.9)])
        assert "tried X, failed" in result

    def test_errors_omitted_when_empty(self):
        t = make_trace(errors=[])
        result = format_traces([(t, 0.9)])
        assert "key errors:" not in result

    def test_multiple_errors_joined(self):
        t = make_trace(errors=["first error", "second error"])
        result = format_traces([(t, 0.9)])
        assert "first error" in result
        assert "second error" in result


class TestMultipleTraces:
    def test_numbered_sequentially(self):
        t1 = make_trace(id="1", task="task one")
        t2 = make_trace(id="2", task="task two")
        result = format_traces([(t1, 0.95), (t2, 0.80)])
        assert "[1]" in result
        assert "[2]" in result

    def test_both_tasks_present(self):
        t1 = make_trace(id="1", task="task one")
        t2 = make_trace(id="2", task="task two")
        result = format_traces([(t1, 0.95), (t2, 0.80)])
        assert "task one" in result
        assert "task two" in result

    def test_header_present_once(self):
        t1 = make_trace(id="1")
        t2 = make_trace(id="2")
        result = format_traces([(t1, 0.95), (t2, 0.80)])
        assert result.count("The following are reasoning traces") == 1
