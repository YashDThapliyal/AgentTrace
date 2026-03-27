"""Tests for SqliteBackend — same contract as JsonlBackend."""
import pytest

from agenttrace.storage.base import Trace
from agenttrace.storage.sqlite import SqliteBackend


@pytest.fixture
def store(tmp_path):
    return SqliteBackend(str(tmp_path / "traces.db"))


def make_trace(suffix: str = "1", **kwargs) -> Trace:
    defaults = dict(
        id=f"id-{suffix}",
        task=f"task {suffix}",
        reasoning=f"reasoning {suffix}",
        outcome=f"outcome {suffix}",
        timestamp="2025-01-01T00:00:00Z",
        embedding=[0.1, 0.2, 0.3],
    )
    defaults.update(kwargs)
    return Trace(**defaults)


class TestBasicCRUD:
    def test_save_and_get_roundtrip(self, store):
        t = make_trace(errors=["err1"], tags=["python"], model="claude", success=False)
        store.save(t)
        got = store.get(t.id)
        assert got.id == t.id
        assert got.task == t.task
        assert got.reasoning == t.reasoning
        assert got.outcome == t.outcome
        assert got.errors == ["err1"]
        assert got.tags == ["python"]
        assert got.model == "claude"
        assert got.success is False
        assert got.timestamp == t.timestamp
        assert got.embedding == pytest.approx(t.embedding)

    def test_list_returns_all(self, store):
        t1, t2 = make_trace("1"), make_trace("2")
        store.save(t1)
        store.save(t2)
        ids = {t.id for t in store.list()}
        assert ids == {"id-1", "id-2"}

    def test_list_empty_store(self, store):
        assert store.list() == []

    def test_delete_removes_trace(self, store):
        t = make_trace()
        store.save(t)
        store.delete(t.id)
        assert all(x.id != t.id for x in store.list())

    def test_delete_unknown_id_raises(self, store):
        with pytest.raises(KeyError):
            store.delete("nonexistent")

    def test_get_unknown_id_raises(self, store):
        with pytest.raises(KeyError):
            store.get("nonexistent")


class TestAllEmbeddings:
    def test_returns_id_embedding_pairs(self, store):
        t1 = make_trace("1", embedding=[1.0, 0.0])
        t2 = make_trace("2", embedding=[0.0, 1.0])
        store.save(t1)
        store.save(t2)
        result = {id_: emb for id_, emb in store.all_embeddings()}
        assert result["id-1"] == pytest.approx([1.0, 0.0])
        assert result["id-2"] == pytest.approx([0.0, 1.0])

    def test_empty_store_returns_empty(self, store):
        assert store.all_embeddings() == []


class TestPersistence:
    def test_traces_persist_across_instances(self, tmp_path):
        path = str(tmp_path / "traces.db")
        store1 = SqliteBackend(path)
        t = make_trace()
        store1.save(t)
        store1.close()

        store2 = SqliteBackend(path)
        assert store2.get(t.id).task == t.task

    def test_file_created_if_missing(self, tmp_path):
        path = tmp_path / "sub" / "deep" / "traces.db"
        store = SqliteBackend(str(path))
        assert path.exists()
        assert store.list() == []


class TestTypeSafety:
    def test_bool_roundtrip(self, store):
        t_true = make_trace("t", success=True)
        t_false = make_trace("f", success=False)
        store.save(t_true)
        store.save(t_false)
        assert store.get("id-t").success is True
        assert store.get("id-f").success is False

    def test_none_model_roundtrip(self, store):
        t = make_trace(model=None)
        store.save(t)
        assert store.get(t.id).model is None

    def test_empty_lists_roundtrip(self, store):
        t = make_trace(errors=[], tags=[], embedding=[])
        store.save(t)
        got = store.get(t.id)
        assert got.errors == []
        assert got.tags == []
        assert got.embedding == []
