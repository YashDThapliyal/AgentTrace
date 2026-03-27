"""Tests for all embedding providers."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agenttrace.config import AgentTraceConfig
from agenttrace.embeddings.base import EmbeddingProvider, get_provider


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_EMBEDDING_384 = [0.1] * 384
_FAKE_EMBEDDING_1536 = [0.1] * 1536
_FAKE_EMBEDDING_512 = [0.1] * 512


class TestEmbeddingProviderABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]


class TestGetProviderFactory:
    def test_unknown_provider_raises(self):
        cfg = AgentTraceConfig(embeddings_provider="unknown")
        with pytest.raises(ValueError, match="unknown"):
            get_provider(cfg)

    def test_returns_local_embedder(self):
        from agenttrace.embeddings.local import LocalEmbedder
        cfg = AgentTraceConfig(embeddings_provider="local")
        provider = get_provider(cfg)
        assert isinstance(provider, LocalEmbedder)

    def test_returns_openai_embedder(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from agenttrace.embeddings.openai import OpenAIEmbedder
        cfg = AgentTraceConfig(embeddings_provider="openai")
        provider = get_provider(cfg)
        assert isinstance(provider, OpenAIEmbedder)

    def test_returns_anthropic_embedder(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from agenttrace.embeddings.anthropic import AnthropicEmbedder
        cfg = AgentTraceConfig(embeddings_provider="anthropic")
        provider = get_provider(cfg)
        assert isinstance(provider, AnthropicEmbedder)


# ---------------------------------------------------------------------------
# LocalEmbedder
# ---------------------------------------------------------------------------

class TestLocalEmbedder:
    def test_embed_returns_list_of_floats(self, monkeypatch):
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: _FAKE_EMBEDDING_384)
        mock_st_cls = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_cls)}):
            from importlib import reload
            import agenttrace.embeddings.local as local_mod
            reload(local_mod)
            embedder = local_mod.LocalEmbedder()
            result = embedder.embed("test text")

        assert isinstance(result, list)
        assert len(result) == 384

    def test_embed_same_input_twice_reuses_model(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: _FAKE_EMBEDDING_384)
        mock_st_cls = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_cls)}):
            from importlib import reload
            import agenttrace.embeddings.local as local_mod
            reload(local_mod)
            embedder = local_mod.LocalEmbedder()
            embedder.embed("first call")
            embedder.embed("second call")

        # Model constructor called once despite two embed() calls
        assert mock_st_cls.call_count == 1

    def test_import_error_raises_helpful_message(self):
        import sys
        # Remove sentence_transformers from modules to force ImportError
        modules_backup = sys.modules.pop("sentence_transformers", None)
        try:
            from importlib import reload
            import agenttrace.embeddings.local as local_mod
            reload(local_mod)
            embedder = local_mod.LocalEmbedder()
            # Force re-load
            embedder._model = None
            with patch.dict("sys.modules", {"sentence_transformers": None}):  # type: ignore[dict-item]
                with pytest.raises(ImportError, match="sentence-transformers"):
                    embedder.embed("test")
        finally:
            if modules_backup is not None:
                sys.modules["sentence_transformers"] = modules_backup


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------

class TestOpenAIEmbedder:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from agenttrace.embeddings.openai import OpenAIEmbedder
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIEmbedder()

    def test_accepts_explicit_api_key(self):
        from agenttrace.embeddings.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(api_key="sk-fake")
        assert embedder._api_key == "sk-fake"

    def test_embed_calls_openai_api(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        from agenttrace.embeddings.openai import OpenAIEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=_FAKE_EMBEDDING_1536)]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            embedder = OpenAIEmbedder(api_key="sk-fake")
            result = embedder.embed("hello world")

        assert result == _FAKE_EMBEDDING_1536
        mock_client.embeddings.create.assert_called_once_with(
            input="hello world", model="text-embedding-3-small"
        )

    def test_missing_openai_package_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        from agenttrace.embeddings.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder(api_key="sk-fake")
        with patch.dict("sys.modules", {"openai": None}):  # type: ignore[dict-item]
            with pytest.raises(ImportError, match="openai"):
                embedder.embed("test")


# ---------------------------------------------------------------------------
# AnthropicEmbedder
# ---------------------------------------------------------------------------

class TestAnthropicEmbedder:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        from agenttrace.embeddings.anthropic import AnthropicEmbedder
        with pytest.raises(ValueError, match="API key"):
            AnthropicEmbedder()

    def test_accepts_anthropic_api_key_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        from agenttrace.embeddings.anthropic import AnthropicEmbedder
        embedder = AnthropicEmbedder()
        assert embedder._api_key == "sk-ant-fake"

    def test_accepts_voyage_api_key_env(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("VOYAGE_API_KEY", "voy-fake")
        from agenttrace.embeddings.anthropic import AnthropicEmbedder
        embedder = AnthropicEmbedder()
        assert embedder._api_key == "voy-fake"

    def test_embed_calls_voyageai(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        from agenttrace.embeddings.anthropic import AnthropicEmbedder

        mock_result = MagicMock()
        mock_result.embeddings = [_FAKE_EMBEDDING_512]
        mock_client = MagicMock()
        mock_client.embed.return_value = mock_result
        mock_voyageai = MagicMock()
        mock_voyageai.Client.return_value = mock_client

        with patch.dict("sys.modules", {"voyageai": mock_voyageai}):
            embedder = AnthropicEmbedder(api_key="sk-fake")
            result = embedder.embed("hello world")

        assert result == _FAKE_EMBEDDING_512
        mock_client.embed.assert_called_once_with(["hello world"], model="voyage-3-lite")

    def test_missing_voyageai_package_raises(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        from agenttrace.embeddings.anthropic import AnthropicEmbedder
        embedder = AnthropicEmbedder(api_key="sk-fake")
        with patch.dict("sys.modules", {"voyageai": None}):  # type: ignore[dict-item]
            with pytest.raises(ImportError, match="voyageai"):
                embedder.embed("test")
