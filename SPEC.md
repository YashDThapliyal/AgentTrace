# AgentTrace — Project Spec v0.1

> Reasoning memory for AI agents. Save what worked. Recall it next time.

---

## What It Is

AgentTrace is a lightweight library that saves successful agent reasoning traces and surfaces them on future runs. When an agent solves a problem, the trace gets stored. The next time a similar problem comes up, the agent gets that context injected — so it doesn't start from zero.

**The core loop:**
```
Task comes in → recall similar past traces → agent runs → save trace if successful
```

---

## Who It's For

Two equal audiences on day one:

| User | How they use it |
|---|---|
| Developer building an agent in Python | `pip install agenttrace`, use the Python API |
| Autonomous agent (Claude Code, Codex, Cursor) | Read `SKILL.md`, call the CLI |

Both share the same underlying core. The interface on top differs.

---

## What a Trace Contains

```json
{
  "id": "uuid",
  "task": "what the agent was asked to do",
  "reasoning": "steps the agent took to solve it",
  "outcome": "what the final result was",
  "errors": ["any errors hit along the way and how they were resolved"],
  "success": true,
  "model": "claude-sonnet-4-5",
  "timestamp": "2025-01-01T00:00:00Z",
  "tags": ["python", "concurrency"],
  "embedding": [0.023, -0.141, "...float array for semantic search"]
}
```

The embedding is generated at `save()` time and stored alongside the trace. It is used at `recall()` time to find semantically similar past traces.

---

## Interfaces

### Python API

```python
import agenttrace

# Before a run — get relevant past traces as a context string
context = agenttrace.recall("fix the threading bug in pipeline.py")

# After a successful run — save the trace
agenttrace.save(
    task="fix the threading bug in pipeline.py",
    reasoning="identified GIL contention, switched threading to multiprocessing...",
    outcome="pipeline now achieves 4x speedup on 8 cores",
    errors=["initially tried thread pool executor, still showed no speedup"],
    success=True,
    tags=["python", "concurrency"]
)
```

### CLI

```bash
# Get relevant past traces for a task (returns formatted context block)
agenttrace recall "fix the threading bug in pipeline.py"

# Save a trace after a successful run
agenttrace save \
  --task "fix the threading bug in pipeline.py" \
  --reasoning "identified GIL contention..." \
  --outcome "pipeline now achieves 4x speedup" \
  --success

# Utilities
agenttrace list                  # list all stored traces
agenttrace inspect <id>          # view a specific trace in full
agenttrace prune --days 30       # delete traces older than N days
agenttrace stats                 # show trace count, storage size, etc.
```

---

## Semantic Search & Retrieval

### Embedding Providers

Embeddings are generated when a trace is saved and used to find similar traces at recall time. Provider is configurable — local by default, no API key required.

| Provider | Default | Requires |
|---|---|---|
| `local` | ✅ Yes | `sentence-transformers` (installed with package) |
| `openai` | No | `OPENAI_API_KEY` env var |
| `anthropic` | No | `ANTHROPIC_API_KEY` env var |

### Retrieval Behavior

- Returns top-k most similar traces
- Filters out any traces below a minimum similarity threshold
- If nothing clears the threshold, returns empty — no garbage context injected

```python
# Defaults: top_k=3, threshold=0.75
context = agenttrace.recall("task", top_k=3, threshold=0.75)
```

Both `top_k` and `threshold` are configurable via config file or function arguments.

### Similarity Algorithm

Cosine similarity against stored embeddings. Run at query time. Fast enough for thousands of traces locally without an index.

---

## Storage Backends

Two backends behind a common interface. Configurable — defaults to JSONL.

### JSONL Backend (default)

- Zero infrastructure
- Human readable, inspectable with any text editor
- Git-friendly — can version control your trace library
- One `.jsonl` file per backend instance

### SQLite Backend

- Better for querying and filtering at scale
- Good for teams sharing a trace store
- Single `.db` file

### Configuration

Resolution order (highest to lowest priority):

1. Explicit argument passed to `agenttrace.init()`
2. `.agenttrace.json` in the current project root
3. `~/.agenttrace/config.json` global config
4. Environment variables (`AGENTTRACE_BACKEND`, `AGENTTRACE_STORE_PATH`)
5. Defaults (`jsonl`, `~/.agenttrace/traces.jsonl`)

Example `.agenttrace.json`:

```json
{
  "backend": "sqlite",
  "store_path": "./.agenttrace/traces.db",
  "embeddings": {
    "provider": "local"
  },
  "recall": {
    "top_k": 3,
    "threshold": 0.75
  }
}
```

---

## SKILL.md (for Claude Code / Codex / Cursor)

A markdown file lives at `skill/SKILL.md`. Autonomous agents read this file at the start of a session to understand what AgentTrace is and how to use it.

It tells the agent:

- What AgentTrace does and why to use it
- **When to call `recall`** — at the start of every task, before doing any work
- **When to call `save`** — immediately after a task succeeds
- The exact CLI commands to run, with examples
- What to do if no traces are returned (proceed normally, don't block)

The SKILL.md is the entire interface for agent users. It must be clear enough that an agent can use AgentTrace correctly with zero prior context.

---

## Project Structure

```
agenttrace/
├── agenttrace/
│   ├── __init__.py              # public API: recall(), save(), init()
│   ├── core.py                  # orchestration — wires storage, embeddings, retrieval
│   ├── config.py                # AgentTraceConfig dataclass + resolution logic
│   ├── storage/
│   │   ├── base.py              # abstract StorageBackend interface
│   │   ├── jsonl.py             # JSONL file backend
│   │   └── sqlite.py            # SQLite backend
│   ├── embeddings/
│   │   ├── base.py              # abstract EmbeddingProvider interface
│   │   ├── local.py             # sentence-transformers (default)
│   │   ├── openai.py            # OpenAI embeddings (optional)
│   │   └── anthropic.py         # Anthropic embeddings (optional)
│   ├── retrieval.py             # cosine similarity + threshold filtering
│   ├── injection.py             # formats retrieved traces into context string
│   └── cli.py                   # CLI entrypoint (click or argparse)
│
├── skill/
│   └── SKILL.md                 # instructions for Claude Code / Codex / Cursor
│
├── examples/
│   ├── python_api_example.py    # basic Python API usage
│   └── claude_code/
│       └── example_session.md   # example of an agent using the CLI skill
│
├── tests/
│   ├── test_core.py
│   ├── test_storage_jsonl.py
│   ├── test_storage_sqlite.py
│   ├── test_embeddings.py
│   └── test_retrieval.py
│
├── README.md
├── pyproject.toml
└── .agenttrace.json.example     # example config file
```

---

## Module Responsibilities

### `__init__.py`
Exposes the public API. Three functions: `init()`, `recall()`, `save()`. Everything else is internal.

### `core.py`
Wires everything together. Holds the `AgentTrace` class that `__init__.py` delegates to. Responsible for:
- Loading config
- Instantiating the right storage backend and embedding provider
- Coordinating the save and recall flows

### `config.py`
`AgentTraceConfig` dataclass. Resolution logic that checks project root → global config → env vars → defaults.

### `storage/base.py`
Abstract interface all backends implement:
```python
class StorageBackend:
    def save(self, trace: Trace) -> None: ...
    def get(self, id: str) -> Trace: ...
    def list(self) -> list[Trace]: ...
    def delete(self, id: str) -> None: ...
    def all_embeddings(self) -> list[tuple[str, list[float]]]: ...
```

### `embeddings/base.py`
Abstract interface all providers implement:
```python
class EmbeddingProvider:
    def embed(self, text: str) -> list[float]: ...
```

### `retrieval.py`
Takes a query embedding and a list of stored embeddings, returns IDs of top-k traces above threshold. Pure function, no side effects.

### `injection.py`
Takes a list of `Trace` objects, returns a formatted string ready to prepend to an agent's context. Format is a structured block the agent can reason about.

### `cli.py`
CLI entrypoint. Commands: `recall`, `save`, `list`, `inspect`, `prune`, `stats`. Uses `click`.

---

## Data Flow

### Save flow
```
agenttrace.save(task, reasoning, outcome, ...) 
  → embed task text via EmbeddingProvider
  → construct Trace object with embedding
  → StorageBackend.save(trace)
```

### Recall flow
```
agenttrace.recall(task_description)
  → embed task_description via EmbeddingProvider
  → StorageBackend.all_embeddings() → list of (id, embedding)
  → retrieval.rank(query_embedding, stored_embeddings, top_k, threshold)
  → StorageBackend.get(id) for each result
  → injection.format(traces) → context string
```

---

## Injection Format

Retrieved traces are formatted into a structured context block for injection:

```
<agent_trace_context>
The following are reasoning traces from similar past tasks.
Use them to inform your approach — especially error patterns and resolutions.

[1] similarity: 0.91 | tags: python, concurrency
task: A data pipeline using Python threads shows no speedup on CPU-bound work
key errors: thread pool executor showed same issue; GIL prevents true parallelism for CPU work
resolution: switched to multiprocessing.Pool — achieved expected speedup

[2] similarity: 0.78 | tags: python, performance
task: Parallel file hashing slower than sequential
key errors: threading module used incorrectly for CPU-bound task
resolution: used concurrent.futures.ProcessPoolExecutor instead of ThreadPoolExecutor
</agent_trace_context>
```

---

## Dependencies

### Required
- `sentence-transformers` — local embeddings (default provider)
- `click` — CLI
- `numpy` — cosine similarity

### Optional
- `openai` — OpenAI embedding provider
- `anthropic` — Anthropic embedding provider

### Dev
- `pytest`
- `ruff`
- `mypy`

---

## Out of Scope for v0.1

These are explicitly deferred. Do not build them now.

- MCP server interface
- Team sync / cloud storage / S3
- Web dashboard or UI
- Trace compression or LLM-based summarization
- Automatic trace quality scoring
- Trace deduplication / clustering
- Staleness / TTL management

---

## Definition of Done

v0.1 is complete when:

- [ ] `pip install agenttrace` works and installs cleanly
- [ ] `agenttrace recall "task"` returns relevant past traces from the CLI
- [ ] `agenttrace save` stores a trace from the CLI
- [ ] `agenttrace.recall()` and `agenttrace.save()` work via the Python API
- [ ] Both JSONL and SQLite backends work and are selectable via config
- [ ] Local embedding provider works out of the box with no API key
- [ ] OpenAI and Anthropic embedding providers work when configured
- [ ] `skill/SKILL.md` is clear enough for Claude Code to use without help
- [ ] `README.md` explains installation, quickstart, and config
- [ ] At least one working example for Python API usage
- [ ] Tests pass for core, storage, embeddings, and retrieval modules
- [ ] `ruff` and `mypy` pass with no errors

---

## Notes for Claude Code Swarm

When building this, work module by module in this order:

1. `config.py` — everything depends on config
2. `storage/base.py` → `storage/jsonl.py` → `storage/sqlite.py`
3. `embeddings/base.py` → `embeddings/local.py` → `embeddings/openai.py` → `embeddings/anthropic.py`
4. `retrieval.py`
5. `injection.py`
6. `core.py` — wires 1-5 together
7. `__init__.py` — thin public API on top of core
8. `cli.py` — CLI on top of core
9. `skill/SKILL.md`
10. `examples/`, `tests/`, `README.md`, `pyproject.toml`

Each module has a clear interface defined above. Build and test each one independently before wiring together.
