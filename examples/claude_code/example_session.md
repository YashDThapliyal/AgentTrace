# Example: Claude Code Session Using AgentTrace

This shows how an autonomous agent (Claude Code) uses AgentTrace across two separate sessions.

---

## Session 1 — Agent solves a task for the first time

**User prompt:** Fix the race condition in `server.py` — requests occasionally return stale data.

**Agent session:**

```bash
# Step 1: Recall past traces before starting
$ agenttrace recall "fix race condition in server.py returning stale data"
No matching traces found.
```

_No past traces — agent proceeds from scratch._

Agent investigates `server.py`, identifies a shared cache dict mutated across threads without a lock, adds `threading.Lock()` around reads and writes, runs the test suite, confirms the fix.

```bash
# Step 2: Save the trace after success
$ agenttrace save \
  --task "fix race condition in server.py returning stale data" \
  --reasoning "Found shared _cache dict mutated by multiple threads without synchronization. Added threading.RLock() — used RLock (reentrant) because get_or_compute() calls itself recursively for nested keys." \
  --outcome "Race condition eliminated. Stress test with 50 concurrent threads passes consistently." \
  --errors "Initially used Lock() — caused deadlock when get_or_compute() was called recursively" \
  --tags "python" --tags "threading" --tags "concurrency" \
  --success

Trace saved: 7f3a9b2c-1234-4abc-8def-fedcba987654
```

---

## Session 2 — Different agent, similar problem

**User prompt:** There's a thread safety issue in `cache_manager.py` — reads sometimes return None.

**Agent session:**

```bash
# Step 1: Recall past traces before starting
$ agenttrace recall "thread safety issue cache_manager.py reads return None"
```

```
<agent_trace_context>
The following are reasoning traces from similar past tasks.
Use them to inform your approach — especially error patterns and resolutions.

[1] similarity: 0.87 | tags: python, threading, concurrency
task: fix race condition in server.py returning stale data
key errors: Initially used Lock() — caused deadlock when get_or_compute() was called recursively
resolution: Race condition eliminated. Stress test with 50 concurrent threads passes consistently.
</agent_trace_context>
```

_Agent reads the trace. Knows to use RLock (not Lock) for recursive cache access patterns._

Agent opens `cache_manager.py`, finds the unprotected cache dict, applies `threading.RLock()` immediately — skipping the Lock → deadlock mistake from the previous session.

```bash
# Step 2: Save the trace after success
$ agenttrace save \
  --task "thread safety issue in cache_manager.py — reads return None" \
  --reasoning "Unprotected shared dict; multiple threads reading during a write. Applied threading.RLock() — used reentrant lock because of recursive cache population pattern (same pattern flagged in past trace)." \
  --outcome "Thread safety issue resolved. 100 concurrent reader threads now return correct values." \
  --tags "python" --tags "threading" --tags "concurrency" \
  --success

Trace saved: a1b2c3d4-5678-90ab-cdef-0123456789ab
```

---

## Key takeaways

- The second agent avoided a known pitfall (Lock vs RLock deadlock) because the first agent's error was saved in the trace.
- `recall` is cheap — run it at the start of every task.
- `save` only runs on success — failed attempts are not stored.
- Over time, the trace store accumulates institutional knowledge that benefits all future agents.
