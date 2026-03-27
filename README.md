
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/d058d5a3-ba3c-4f53-b48b-22911b3dd933" />

# AgentTrace
> Reasoning memory for AI agents. Save what worked. Recall it next time.

AgentTrace is a lightweight Python library that saves successful agent reasoning traces and surfaces them on future runs. When an agent solves a problem, the trace gets stored. The next time a similar problem comes up, the agent gets that context injected — so it doesn't start from zero.

**The core loop:**
```
Task comes in → recall similar past traces → agent runs → save trace if successful
```

---

## Installation

```bash
pip install agenttrace
```

Optional embedding providers:

```bash
pip install agenttrace[openai]      # OpenAI embeddings
pip install agenttrace[anthropic]   # Voyage AI / Anthropic embeddings
```

---

## Quickstart

### Python API

```python
import agenttrace

# Before a run — get relevant past traces as a context string
context = agenttrace.recall("fix the threading bug in pipeline.py")

# ... run your agent with the context ...

# After a successful run — save the trace
agenttrace.save(
    task="fix the threading bug in pipeline.py",
    reasoning="identified GIL contention, switched threading to multiprocessing...",
    outcome="pipeline now achieves 4x speedup on 8 cores",
    errors=["initially tried thread pool executor, still showed no speedup"],
    success=True,
    tags=["python", "concurrency"],
)
```

### CLI (for autonomous agents)

```bash
# Recall relevant past traces
agenttrace recall "fix the threading bug in pipeline.py"

# Save a trace after a successful run
agenttrace save \
  --task "fix the threading bug in pipeline.py" \
  --reasoning "identified GIL contention..." \
  --outcome "pipeline now achieves 4x speedup" \
  --errors "initially tried thread pool executor, still no speedup" \
  --tags "python" --tags "concurrency" \
  --success

# Other utilities
agenttrace list                  # list all stored traces
agenttrace inspect <id>          # view a specific trace in full
agenttrace prune --days 30       # delete traces older than 30 days
agenttrace stats                 # show trace count, storage size, etc.
```

---

## Configuration

AgentTrace resolves configuration in this priority order (highest → lowest):

1. Explicit `agenttrace.init()` arguments
2. `.agenttrace.json` in the current project directory
3. `~/.agenttrace/config.json` (global config)
4. Environment variables (`AGENTTRACE_BACKEND`, `AGENTTRACE_STORE_PATH`, etc.)
5. Defaults

Example `.agenttrace.json`:

```json
{
  "backend": "sqlite",
  "store_path": "./.agenttrace/traces.db",
  "embeddings_provider": "local",
  "recall": {
    "top_k": 3,
    "threshold": 0.5
  }
}
```

### Embedding Providers

| Provider | Default | Requires |
|---|---|---|
| `local` | ✅ Yes | `sentence-transformers` (installed with package) |
| `openai` | No | `OPENAI_API_KEY` env var |
| `anthropic` | No | `ANTHROPIC_API_KEY` or `VOYAGE_API_KEY` env var |

### Storage Backends

| Backend | Default | Description |
|---|---|---|
| `jsonl` | ✅ Yes | One `.jsonl` file — human-readable, git-friendly |
| `sqlite` | No | SQLite `.db` file — better for scale and filtering |

---

## For Autonomous Agents (Claude Code / Codex / Cursor)

See [`skill/SKILL.md`](skill/SKILL.md) for complete instructions. The short version:

1. Run `agenttrace recall "your task"` at the start of every task.
2. Do your work.
3. Run `agenttrace save ...` immediately after the task succeeds.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check .
mypy agenttrace/
```

---

## What a Trace Contains

```json
{
  "id": "uuid",
  "task": "what the agent was asked to do",
  "reasoning": "steps the agent took to solve it",
  "outcome": "what the final result was",
  "errors": ["errors hit along the way and how they were resolved"],
  "success": true,
  "model": "claude-sonnet-4-6",
  "timestamp": "2025-01-01T00:00:00Z",
  "tags": ["python", "concurrency"],
  "embedding": [0.023, -0.141, "..."]
}
```

---

## Out of Scope for v0.1

- MCP server interface
- Team sync / cloud storage
- Web dashboard or UI
- Trace compression or LLM-based summarization
- Automatic trace quality scoring
- Trace deduplication / clustering
