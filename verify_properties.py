"""
Targeted verification of AgentTrace logical correctness.
Covers: embedding, retrieval, storage, injection, config.
Each check prints exact output and PASS/FAIL.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agenttrace.config import AgentTraceConfig, load_config
from agenttrace.core import AgentTrace
from agenttrace.injection import format_traces
from agenttrace.retrieval import rank
from agenttrace.storage.base import Trace
from agenttrace.storage.jsonl import JsonlBackend
from agenttrace.storage.sqlite import SqliteBackend

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {'✓' if condition else '✗'} {label}: {status}{suffix}")
    if not condition:
        _failures.append(label)
    return condition


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ─────────────────────────────────────────────────────────────
# 1. EMBEDDING CORRECTNESS
# ─────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
print("1. EMBEDDING CORRECTNESS")
print("══════════════════════════════════════════════")

try:
    import warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sentence_transformers import SentenceTransformer
    print("  Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    task_a = "fix python threading performance bug"
    task_b = "debug concurrency issue with Python threads"
    task_c = "bake a chocolate soufflé from scratch"

    emb_a = model.encode(task_a).tolist()
    emb_b = model.encode(task_b).tolist()
    emb_c = model.encode(task_c).tolist()

    sim_similar   = cosine(emb_a, emb_b)
    sim_unrelated = cosine(emb_a, emb_c)

    print(f"\n  Pair A: '{task_a}'")
    print(f"  Pair B: '{task_b}'")
    print(f"  → cosine similarity: {sim_similar:.4f}")
    check("similar tasks > 0.5", sim_similar > 0.5, f"{sim_similar:.4f}")

    print(f"\n  Pair A: '{task_a}'")
    print(f"  Pair C: '{task_c}'")
    print(f"  → cosine similarity: {sim_unrelated:.4f}")
    check("unrelated tasks < 0.3", sim_unrelated < 0.3, f"{sim_unrelated:.4f}")

except ImportError:
    print(f"  ✗ sentence-transformers not installed: {FAIL}")
    _failures.append("sentence-transformers import")


# ─────────────────────────────────────────────────────────────
# 2. RETRIEVAL CORRECTNESS
# ─────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
print("2. RETRIEVAL CORRECTNESS")
print("══════════════════════════════════════════════")


# 2a. Results returned in descending similarity order
stored = [
    ("low",  [0.0, 1.0, 0.0]),
    ("high", [1.0, 0.0, 0.0]),
    ("mid",  [0.8, 0.2, 0.0]),
]
result = rank([1.0, 0.0, 0.0], stored, top_k=10, threshold=0.0)
scores = [s for _, s in result]
print(f"\n  Scores returned: {[round(s, 4) for s in scores]}")
check("descending similarity order", scores == sorted(scores, reverse=True))

# 2b. Nothing below threshold is ever returned
stored_thresh = [
    ("above", [1.0, 0.0]),   # similarity 1.0 → passes 0.7
    ("below", [0.0, 1.0]),   # similarity 0.0 → fails 0.7
]
result_thresh = rank([1.0, 0.0], stored_thresh, top_k=5, threshold=0.7)
ids_thresh = [i for i, _ in result_thresh]
print(f"\n  Threshold=0.7, IDs returned: {ids_thresh}")
check("nothing below threshold returned",
      "above" in ids_thresh and "below" not in ids_thresh)

# 2c. top_k=3 with 5 traces above threshold → exactly 3 returned
stored_5 = [(f"id{i}", [1.0 - i*0.05, i*0.05]) for i in range(5)]
result_topk = rank([1.0, 0.0], stored_5, top_k=3, threshold=0.0)
print(f"\n  5 traces above threshold, top_k=3 → {len(result_topk)} returned")
check("top_k=3 with 5 candidates → 3 results", len(result_topk) == 3)

# 2d. Nothing above threshold → empty list (core turns this into "")

with tempfile.TemporaryDirectory() as tmp:
    cfg = AgentTraceConfig(backend="jsonl",
                           store_path=f"{tmp}/t.jsonl",
                           threshold=0.99)
    at = AgentTrace(cfg)
    at._embedder = MagicMock()
    at._embedder.embed.return_value = [1.0, 0.0]
    at.save(task="t", reasoning="r", outcome="o")
    # query vector is orthogonal → similarity 0.0, below threshold 0.99
    at._embedder.embed.return_value = [0.0, 1.0]
    result_empty = at.recall("unrelated")

print(f"\n  Recall with no match above threshold → {repr(result_empty)}")
check("nothing above threshold → empty string (not error)",
      result_empty == "")


# ─────────────────────────────────────────────────────────────
# 3. STORAGE CORRECTNESS
# ─────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
print("3. STORAGE CORRECTNESS")
print("══════════════════════════════════════════════")


FIXED_EMBEDDING = [0.1, 0.25, -0.5, 0.99, 0.0]

# 3a. Saved trace retrievable by ID with all fields intact (JSONL)
with tempfile.TemporaryDirectory() as tmp:
    store = JsonlBackend(f"{tmp}/traces.jsonl")
    original = Trace(
        id="verify-001",
        task="fix threading bug",
        reasoning="use multiprocessing",
        outcome="4x speedup",
        errors=["tried thread pool"],
        success=True,
        model="claude-sonnet-4-6",
        timestamp="2026-01-01T00:00:00Z",
        tags=["python", "concurrency"],
        embedding=FIXED_EMBEDDING,
    )
    store.save(original)
    got = store.get("verify-001")

    print("\n  JSONL round-trip fields:")
    print(f"    id={got.id}, task='{got.task}', errors={got.errors}, tags={got.tags}")
    check("JSONL: all fields intact",
          got.id == original.id and
          got.task == original.task and
          got.reasoning == original.reasoning and
          got.outcome == original.outcome and
          got.errors == original.errors and
          got.success == original.success and
          got.model == original.model and
          got.tags == original.tags)

    # 3b. Embedding stored is identical to the one computed at save time
    print(f"\n  Stored embedding:   {got.embedding}")
    print(f"  Original embedding: {FIXED_EMBEDDING}")
    check("JSONL: stored embedding identical to computed",
          got.embedding == FIXED_EMBEDDING)

# 3c. JSONL and SQLite produce identical results for the same input
with tempfile.TemporaryDirectory() as tmp:
    jsonl_store = JsonlBackend(f"{tmp}/traces.jsonl")
    sqlite_store = SqliteBackend(f"{tmp}/traces.db")

    for store in (jsonl_store, sqlite_store):
        store.save(Trace(
            id="cmp-001",
            task="compare backends",
            reasoning="save same trace to both",
            outcome="should be identical",
            errors=["err1", "err2"],
            success=True,
            model="test-model",
            timestamp="2026-06-01T00:00:00Z",
            tags=["test"],
            embedding=[0.1, 0.2, 0.3],
        ))

    j = jsonl_store.get("cmp-001")
    s = sqlite_store.get("cmp-001")

    fields_match = (
        j.id == s.id and
        j.task == s.task and
        j.reasoning == s.reasoning and
        j.outcome == s.outcome and
        j.errors == s.errors and
        j.success == s.success and
        j.model == s.model and
        j.timestamp == s.timestamp and
        j.tags == s.tags and
        all(abs(a - b) < 1e-9 for a, b in zip(j.embedding, s.embedding))
    )
    print(f"\n  JSONL embedding:  {j.embedding}")
    print(f"  SQLite embedding: {s.embedding}")
    check("JSONL and SQLite produce identical results", fields_match)


# ─────────────────────────────────────────────────────────────
# 4. INJECTION CORRECTNESS
# ─────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
print("4. INJECTION CORRECTNESS")
print("══════════════════════════════════════════════")


t = Trace(
    id="inj-1",
    task="fix the threading bug",
    reasoning="switched to multiprocessing",
    outcome="achieved 4x speedup",
    errors=["ThreadPoolExecutor showed no improvement"],
    success=True,
    model=None,
    timestamp="2026-01-01T00:00:00Z",
    tags=["python", "concurrency"],
    embedding=[],
)
formatted = format_traces([(t, 0.87)])

print("\n  Formatted output:\n")
for line in formatted.splitlines():
    print(f"    {line}")

check("\ncontains similarity score", "similarity: 0.87" in formatted)
check("contains tags",              "python" in formatted and "concurrency" in formatted)
check("contains task",              "fix the threading bug" in formatted)
check("contains errors",            "ThreadPoolExecutor showed no improvement" in formatted)
check("contains resolution",        "achieved 4x speedup" in formatted)
check("wrapped in XML tags",        formatted.startswith("<agent_trace_context>") and
                                    formatted.strip().endswith("</agent_trace_context>"))

# Empty result → empty string
empty_result = format_traces([])
print(f"\n  format_traces([]) → {repr(empty_result)}")
check("empty input → empty string (not partial block)", empty_result == "")


# ─────────────────────────────────────────────────────────────
# 5. CONFIG CORRECTNESS
# ─────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
print("5. CONFIG CORRECTNESS")
print("══════════════════════════════════════════════")


# 5a. Default threshold is 0.5
default_cfg = AgentTraceConfig()
print(f"\n  AgentTraceConfig() default threshold: {default_cfg.threshold}")
check("default threshold is 0.5", default_cfg.threshold == 0.5)

# 5b. Project config overrides global config
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    global_cfg_path = tmp_path / "global_config.json"
    global_cfg_path.write_text(json.dumps({"threshold": 0.8, "top_k": 10}))

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".agenttrace.json").write_text(json.dumps({"threshold": 0.6}))

    orig_global = __import__("agenttrace.config", fromlist=["config"])._GLOBAL_CONFIG_PATH
    import agenttrace.config as cfg_mod
    cfg_mod._GLOBAL_CONFIG_PATH = global_cfg_path
    os.chdir(project_dir)

    loaded = load_config()
    cfg_mod._GLOBAL_CONFIG_PATH = orig_global
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(f"\n  global threshold=0.8, project threshold=0.6 → loaded: {loaded.threshold}")
    print(f"  global top_k=10 (no project override) → loaded top_k: {loaded.top_k}")
    check("project config overrides global config", loaded.threshold == 0.6)
    check("global config used when project doesn't override", loaded.top_k == 10)

# 5c. Env vars override file config
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    project_dir = tmp_path / "p"
    project_dir.mkdir()
    (project_dir / ".agenttrace.json").write_text(json.dumps({"threshold": 0.6}))

    os.environ["AGENTTRACE_THRESHOLD"] = "0.42"
    os.chdir(project_dir)

    loaded_env = load_config()
    del os.environ["AGENTTRACE_THRESHOLD"]
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(f"\n  project threshold=0.6, AGENTTRACE_THRESHOLD=0.42 → loaded: {loaded_env.threshold}")
    check("env var overrides file config", abs(loaded_env.threshold - 0.42) < 1e-9)


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
print("SUMMARY")
print("══════════════════════════════════════════════")
if _failures:
    print(f"\n  {len(_failures)} check(s) FAILED:")
    for f in _failures:
        print(f"    ✗ {f}")
    sys.exit(1)
else:
    print("\n  All checks PASSED.")
    sys.exit(0)
