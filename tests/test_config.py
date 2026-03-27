"""Tests for agenttrace.config — AgentTraceConfig and load_config()."""
import json
import os
from pathlib import Path

import pytest

from agenttrace.config import AgentTraceConfig, load_config


class TestDefaults:
    def test_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENTTRACE_BACKEND", raising=False)
        monkeypatch.delenv("AGENTTRACE_STORE_PATH", raising=False)
        monkeypatch.delenv("AGENTTRACE_EMBEDDINGS_PROVIDER", raising=False)
        monkeypatch.delenv("AGENTTRACE_TOP_K", raising=False)
        monkeypatch.delenv("AGENTTRACE_THRESHOLD", raising=False)
        # Override global config path to a nonexistent location
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        cfg = load_config()

        assert cfg.backend == "jsonl"
        assert cfg.embeddings_provider == "local"
        assert cfg.top_k == 3
        assert cfg.threshold == 0.75
        assert cfg.store_path.endswith("traces.jsonl")


class TestEnvVarOverrides:
    def test_backend_env_var(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENTTRACE_BACKEND", "sqlite")
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        cfg = load_config()
        assert cfg.backend == "sqlite"

    def test_store_path_env_var(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENTTRACE_STORE_PATH", "/tmp/my-traces.jsonl")
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        cfg = load_config()
        assert cfg.store_path == "/tmp/my-traces.jsonl"

    def test_embeddings_provider_env_var(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENTTRACE_EMBEDDINGS_PROVIDER", "openai")
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        cfg = load_config()
        assert cfg.embeddings_provider == "openai"

    def test_top_k_env_var(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENTTRACE_TOP_K", "5")
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        cfg = load_config()
        assert cfg.top_k == 5

    def test_threshold_env_var(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENTTRACE_THRESHOLD", "0.8")
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        cfg = load_config()
        assert cfg.threshold == 0.8

    def test_invalid_top_k_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENTTRACE_TOP_K", "abc")
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        with pytest.raises(ValueError, match="AGENTTRACE_TOP_K"):
            load_config()

    def test_invalid_threshold_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENTTRACE_THRESHOLD", "not-a-float")
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        with pytest.raises(ValueError, match="AGENTTRACE_THRESHOLD"):
            load_config()


class TestProjectConfigFile:
    def test_project_config_overrides_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENTTRACE_BACKEND", raising=False)
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        (tmp_path / ".agenttrace.json").write_text(json.dumps({
            "backend": "sqlite",
            "store_path": "/tmp/project.db",
        }))

        cfg = load_config()
        assert cfg.backend == "sqlite"
        assert cfg.store_path == "/tmp/project.db"
        assert cfg.top_k == 3  # still default

    def test_project_config_overrides_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENTTRACE_BACKEND", "jsonl")
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        (tmp_path / ".agenttrace.json").write_text(json.dumps({"backend": "sqlite"}))

        cfg = load_config()
        assert cfg.backend == "sqlite"

    def test_malformed_project_config_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")
        (tmp_path / ".agenttrace.json").write_text("{not valid json")

        with pytest.raises(ValueError, match=".agenttrace.json"):
            load_config()

    def test_unknown_keys_ignored(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")
        (tmp_path / ".agenttrace.json").write_text(json.dumps({
            "backend": "sqlite",
            "unknown_future_key": "ignored",
        }))

        cfg = load_config()
        assert cfg.backend == "sqlite"


class TestGlobalConfigFile:
    def test_global_config_overrides_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENTTRACE_BACKEND", raising=False)

        global_cfg = tmp_path / "global-config.json"
        global_cfg.write_text(json.dumps({"backend": "sqlite", "top_k": 10}))
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", global_cfg)

        cfg = load_config()
        assert cfg.backend == "sqlite"
        assert cfg.top_k == 10

    def test_project_config_beats_global(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        global_cfg = tmp_path / "global-config.json"
        global_cfg.write_text(json.dumps({"backend": "sqlite"}))
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", global_cfg)

        (tmp_path / ".agenttrace.json").write_text(json.dumps({"backend": "jsonl"}))

        cfg = load_config()
        assert cfg.backend == "jsonl"

    def test_malformed_global_config_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        global_cfg = tmp_path / "global-config.json"
        global_cfg.write_text("{bad json")
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", global_cfg)

        with pytest.raises(ValueError):
            load_config()


class TestExplicitOverrides:
    def test_explicit_overrides_beat_all(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENTTRACE_BACKEND", "sqlite")
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")
        (tmp_path / ".agenttrace.json").write_text(json.dumps({"backend": "sqlite"}))

        cfg = load_config(backend="jsonl")
        assert cfg.backend == "jsonl"

    def test_explicit_partial_overrides(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        cfg = load_config(top_k=7, threshold=0.5)
        assert cfg.top_k == 7
        assert cfg.threshold == 0.5
        assert cfg.backend == "jsonl"  # still default


class TestStorePath:
    def test_tilde_expanded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", tmp_path / "no-config.json")

        cfg = load_config(store_path="~/traces.jsonl")
        assert not cfg.store_path.startswith("~")
        assert cfg.store_path == str(Path("~/traces.jsonl").expanduser())


class TestPriorityOrdering:
    def test_full_priority_chain(self, tmp_path, monkeypatch):
        """explicit > project > global > env > defaults"""
        monkeypatch.chdir(tmp_path)

        # env var
        monkeypatch.setenv("AGENTTRACE_TOP_K", "1")

        # global config
        global_cfg = tmp_path / "global-config.json"
        global_cfg.write_text(json.dumps({"top_k": 2}))
        monkeypatch.setattr("agenttrace.config._GLOBAL_CONFIG_PATH", global_cfg)

        # project config
        (tmp_path / ".agenttrace.json").write_text(json.dumps({"top_k": 3}))

        # explicit
        cfg = load_config(top_k=4)
        assert cfg.top_k == 4

        # project beats global and env
        cfg2 = load_config()
        assert cfg2.top_k == 3
