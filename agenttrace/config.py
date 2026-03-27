"""Configuration loading for AgentTrace.

Resolution order (highest → lowest priority):
  1. Explicit kwargs passed to load_config()
  2. .agenttrace.json in current working directory
  3. ~/.agenttrace/config.json (global config)
  4. Environment variables (AGENTTRACE_*)
  5. Dataclass defaults
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Exposed as a module-level variable so tests can monkeypatch it.
_GLOBAL_CONFIG_PATH: Path = Path.home() / ".agenttrace" / "config.json"

_PROJECT_CONFIG_NAME = ".agenttrace.json"

_ENV_MAP: dict[str, str] = {
    "backend": "AGENTTRACE_BACKEND",
    "store_path": "AGENTTRACE_STORE_PATH",
    "embeddings_provider": "AGENTTRACE_EMBEDDINGS_PROVIDER",
    "top_k": "AGENTTRACE_TOP_K",
    "threshold": "AGENTTRACE_THRESHOLD",
}

_KNOWN_KEYS = {"backend", "store_path", "embeddings_provider", "top_k", "threshold"}


@dataclass
class AgentTraceConfig:
    backend: str = "jsonl"
    store_path: str = field(
        default_factory=lambda: str(Path.home() / ".agenttrace" / "traces.jsonl")
    )
    embeddings_provider: str = "local"
    top_k: int = 3
    threshold: float = 0.75


def _read_json_file(path: Path, label: str) -> dict[str, object]:
    """Read a JSON config file, returning only known keys. Raises ValueError on bad JSON."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {exc}") from exc
    return {k: v for k, v in raw.items() if k in _KNOWN_KEYS}


def _read_env() -> dict[str, object]:
    """Read known env vars, coercing types."""
    result: dict[str, object] = {}
    for field_name, env_name in _ENV_MAP.items():
        val = os.environ.get(env_name)
        if val is None:
            continue
        if field_name == "top_k":
            try:
                result[field_name] = int(val)
            except ValueError:
                raise ValueError(
                    f"{env_name} must be an integer, got: {val!r}"
                ) from None
        elif field_name == "threshold":
            try:
                result[field_name] = float(val)
            except ValueError:
                raise ValueError(
                    f"{env_name} must be a float, got: {val!r}"
                ) from None
        else:
            result[field_name] = val
    return result


def load_config(**overrides: object) -> AgentTraceConfig:
    """Load config by merging all sources in priority order.

    Keyword arguments override everything else.
    """
    # Start with defaults
    merged: dict[str, object] = {}

    # Layer 4: env vars
    merged.update(_read_env())

    # Layer 3: global config (~/.agenttrace/config.json)
    global_path = _GLOBAL_CONFIG_PATH
    if isinstance(global_path, Path) and global_path.exists():
        merged.update(_read_json_file(global_path, str(global_path)))

    # Layer 2: project config (.agenttrace.json in cwd)
    project_path = Path.cwd() / _PROJECT_CONFIG_NAME
    if project_path.exists():
        merged.update(_read_json_file(project_path, _PROJECT_CONFIG_NAME))

    # Layer 1: explicit overrides
    merged.update({k: v for k, v in overrides.items() if k in _KNOWN_KEYS})

    # Build config, expanding ~ in store_path
    cfg = AgentTraceConfig(**{k: v for k, v in merged.items()})  # type: ignore[arg-type]
    cfg.store_path = str(Path(cfg.store_path).expanduser())
    return cfg
