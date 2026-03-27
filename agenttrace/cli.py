"""CLI entrypoint for AgentTrace.

Commands: recall, save, list, inspect, prune, stats
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import click

import agenttrace as _api
from agenttrace.core import AgentTrace


def _get_core() -> AgentTrace:
    """Return the module-level singleton with stderr status output enabled."""
    instance = _api._get_instance()
    instance._status_io = sys.stderr
    return instance


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """AgentTrace — reasoning memory for AI agents."""


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("task")
@click.option("--top-k", type=int, default=None, help="Max number of results.")
@click.option("--threshold", type=float, default=None, help="Min similarity score.")
def recall(task: str, top_k: int | None, threshold: float | None) -> None:
    """Get relevant past traces for TASK."""
    result = _get_core().recall(task, top_k=top_k, threshold=threshold)
    if result:
        click.echo(result)
    else:
        click.echo("No matching traces found.")


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--task", required=True, help="Task description.")
@click.option("--reasoning", required=True, help="Steps taken to solve it.")
@click.option("--outcome", required=True, help="Final result.")
@click.option("--errors", multiple=True, help="Errors encountered (repeat for multiple).")
@click.option("--model", default=None, help="Model used.")
@click.option("--tags", multiple=True, help="Tags (repeat for multiple).")
@click.option("--success/--no-success", default=True, help="Whether the task succeeded.")
def save(
    task: str,
    reasoning: str,
    outcome: str,
    errors: tuple[str, ...],
    model: str | None,
    tags: tuple[str, ...],
    success: bool,
) -> None:
    """Save a trace after a successful run."""
    trace_id = _get_core().save(
        task=task,
        reasoning=reasoning,
        outcome=outcome,
        errors=list(errors),
        success=success,
        model=model,
        tags=list(tags),
    )
    click.echo(f"Trace saved: {trace_id}")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@cli.command(name="list")
def list_traces() -> None:
    """List all stored traces."""
    traces = _get_core()._get_storage().list()
    if not traces:
        click.echo("No traces stored yet.")
        return
    for trace in traces:
        tags_str = f"  [{', '.join(trace.tags)}]" if trace.tags else ""
        task_preview = trace.task[:60] + ("…" if len(trace.task) > 60 else "")
        click.echo(f"{trace.id[:8]}  {trace.timestamp}  {task_preview}{tags_str}")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("id")
def inspect(id: str) -> None:
    """Show full details of trace ID."""
    try:
        trace = _get_core()._get_storage().get(id)
    except KeyError:
        click.echo(f"Error: trace '{id}' not found.", err=True)
        sys.exit(1)
    click.echo(json.dumps(dataclasses.asdict(trace), indent=2))


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--days", required=True, type=int, help="Delete traces older than N days.")
def prune(days: int) -> None:
    """Delete traces older than DAYS days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    storage = _get_core()._get_storage()
    traces = storage.list()
    to_delete = [
        t for t in traces
        if datetime.fromisoformat(t.timestamp.replace("Z", "+00:00")) < cutoff
    ]
    if not to_delete:
        click.echo("No traces to prune.")
        return
    for trace in to_delete:
        storage.delete(trace.id)
    click.echo(f"Pruned {len(to_delete)} trace{'s' if len(to_delete) != 1 else ''}.")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@cli.command()
def stats() -> None:
    """Show trace count and storage info."""
    core = _get_core()
    storage = core._get_storage()
    traces = storage.list()
    count = len(traces)

    store_path = core._config.store_path
    try:
        size = os.path.getsize(store_path)
        size_str = _human_size(size)
    except OSError:
        size_str = "unknown"

    click.echo(f"Traces:   {count}")
    click.echo(f"Storage:  {store_path} ({size_str})")
    click.echo(f"Provider: {core._config.embeddings_provider}")

    if traces:
        timestamps = sorted(t.timestamp for t in traces)
        click.echo(f"Oldest:   {timestamps[0]}")
        click.echo(f"Newest:   {timestamps[-1]}")


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size //= 1024
    return f"{size:.1f} TB"
