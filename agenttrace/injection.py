"""Format retrieved traces into a context block for agent injection."""
from __future__ import annotations

from agenttrace.storage.base import Trace

_HEADER = (
    "The following are reasoning traces from similar past tasks.\n"
    "Use them to inform your approach — especially error patterns and resolutions."
)


def format_traces(traces_with_scores: list[tuple[Trace, float]]) -> str:
    """Format a list of (Trace, similarity_score) pairs into an XML context block.

    Returns an empty string if the input list is empty.
    """
    if not traces_with_scores:
        return ""

    lines: list[str] = ["<agent_trace_context>", _HEADER, ""]

    for i, (trace, score) in enumerate(traces_with_scores, start=1):
        header = f"[{i}] similarity: {score:.2f}"
        if trace.tags:
            header += f" | tags: {', '.join(trace.tags)}"
        lines.append(header)
        lines.append(f"task: {trace.task}")
        if trace.errors:
            lines.append(f"key errors: {'; '.join(trace.errors)}")
        lines.append(f"resolution: {trace.outcome}")
        lines.append("")

    lines.append("</agent_trace_context>")
    return "\n".join(lines)
