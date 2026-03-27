"""JSONL file storage backend."""
from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

from agenttrace.storage.base import StorageBackend, Trace


class JsonlBackend(StorageBackend):
    """Stores traces as newline-delimited JSON (one trace per line).

    Human-readable, git-friendly, zero infrastructure.
    Not safe for concurrent writes (v0.1 limitation).
    """

    def __init__(self, store_path: str) -> None:
        self._path = Path(store_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    def _read_all(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for line_no, line in enumerate(self._path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                warnings.warn(
                    f"Skipping malformed JSON on line {line_no} of {self._path}",
                    stacklevel=2,
                )
        return records

    def _write_all(self, records: list[dict[str, object]]) -> None:
        if records:
            self._path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        else:
            self._path.write_text("")

    @staticmethod
    def _to_dict(trace: Trace) -> dict[str, object]:
        return dataclasses.asdict(trace)

    @staticmethod
    def _from_dict(data: dict[str, object]) -> Trace:
        raw_errors = data.get("errors") or []
        raw_tags = data.get("tags") or []
        raw_embedding = data.get("embedding") or []
        return Trace(
            id=str(data["id"]),
            task=str(data["task"]),
            reasoning=str(data["reasoning"]),
            outcome=str(data["outcome"]),
            timestamp=str(data["timestamp"]),
            errors=[str(e) for e in raw_errors],  # type: ignore[attr-defined]
            success=bool(data.get("success", True)),
            model=str(data["model"]) if data.get("model") is not None else None,
            tags=[str(t) for t in raw_tags],  # type: ignore[attr-defined]
            embedding=[float(v) for v in raw_embedding],  # type: ignore[attr-defined]
        )

    def save(self, trace: Trace) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps(self._to_dict(trace)) + "\n")

    def get(self, id: str) -> Trace:
        for record in self._read_all():
            if record.get("id") == id:
                return self._from_dict(record)
        raise KeyError(id)

    def delete(self, id: str) -> None:
        records = self._read_all()
        remaining = [r for r in records if r.get("id") != id]
        if len(remaining) == len(records):
            raise KeyError(id)
        self._write_all(remaining)

    def all_embeddings(self) -> list[tuple[str, list[float]]]:
        result: list[tuple[str, list[float]]] = []
        for record in self._read_all():
            trace_id = record.get("id")
            embedding = record.get("embedding")
            if isinstance(trace_id, str) and isinstance(embedding, list):
                result.append((trace_id, [float(v) for v in embedding]))
        return result

    def list(self) -> list[Trace]:
        return [self._from_dict(r) for r in self._read_all()]
