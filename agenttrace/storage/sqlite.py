"""SQLite storage backend."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agenttrace.storage.base import StorageBackend, Trace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id          TEXT PRIMARY KEY,
    task        TEXT NOT NULL,
    reasoning   TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    errors      TEXT NOT NULL,
    success     INTEGER NOT NULL,
    model       TEXT,
    timestamp   TEXT NOT NULL,
    tags        TEXT NOT NULL,
    embedding   TEXT NOT NULL
)
"""


class SqliteBackend(StorageBackend):
    """Stores traces in a SQLite database.

    Better for querying and filtering at scale.
    Thread safety: check_same_thread=False (v0.1 limitation — no row locking).
    """

    def __init__(self, store_path: str) -> None:
        path = Path(store_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    @staticmethod
    def _row_to_trace(row: tuple[object, ...]) -> Trace:
        (id_, task, reasoning, outcome, errors_json, success_int,
         model, timestamp, tags_json, embedding_json) = row
        return Trace(
            id=str(id_),
            task=str(task),
            reasoning=str(reasoning),
            outcome=str(outcome),
            errors=json.loads(str(errors_json)),
            success=bool(success_int),
            model=str(model) if model is not None else None,
            timestamp=str(timestamp),
            tags=json.loads(str(tags_json)),
            embedding=json.loads(str(embedding_json)),
        )

    @staticmethod
    def _trace_to_row(trace: Trace) -> tuple[object, ...]:
        return (
            trace.id,
            trace.task,
            trace.reasoning,
            trace.outcome,
            json.dumps(trace.errors),
            int(trace.success),
            trace.model,
            trace.timestamp,
            json.dumps(trace.tags),
            json.dumps(trace.embedding),
        )

    def save(self, trace: Trace) -> None:
        self._conn.execute(
            "INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?,?)",
            self._trace_to_row(trace),
        )
        self._conn.commit()

    def get(self, id: str) -> Trace:
        cur = self._conn.execute("SELECT * FROM traces WHERE id = ?", (id,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(id)
        return self._row_to_trace(row)

    def delete(self, id: str) -> None:
        cur = self._conn.execute("DELETE FROM traces WHERE id = ?", (id,))
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(id)

    def all_embeddings(self) -> list[tuple[str, list[float]]]:
        cur = self._conn.execute("SELECT id, embedding FROM traces")
        return [(row[0], json.loads(row[1])) for row in cur.fetchall()]

    def list(self) -> list[Trace]:
        cur = self._conn.execute("SELECT * FROM traces")
        return [self._row_to_trace(row) for row in cur.fetchall()]
