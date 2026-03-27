# AgentTrace Skill

## What is AgentTrace

AgentTrace gives you reasoning memory across sessions. When you successfully solve a task, you save the trace. At the start of your next similar task, you recall relevant past traces and get that context injected — so you don't start from zero.

## When to Call `recall`

**At the very start of every task, before doing any work.**

Run `agenttrace recall` with the task description. If matching traces are returned, read them carefully — especially the error patterns and resolutions. They may save you significant time.

## When to Call `save`

**Immediately after a task succeeds.**

Save the trace while the details are fresh. Be specific in `--reasoning` and `--outcome`. Include any errors you hit along the way in `--errors` — these are the most valuable part for future recall.

Do NOT save on failure. Only save when the task is complete and the outcome is confirmed.

## CLI Commands

### Recall past traces
```bash
agenttrace recall "your task description here"
```

Example:
```bash
agenttrace recall "fix the threading bug in pipeline.py"
```

If relevant traces exist, you'll receive a structured context block to inform your approach. If nothing is returned, proceed normally — no traces exist yet for this type of task.

### Save a trace after success
```bash
agenttrace save \
  --task "your task description" \
  --reasoning "the steps you took to solve it" \
  --outcome "what the final result was" \
  --errors "error you hit along the way" \
  --errors "another error (repeat --errors for multiple)" \
  --tags "python" --tags "concurrency" \
  --success
```

Example:
```bash
agenttrace save \
  --task "fix the threading bug in pipeline.py" \
  --reasoning "identified GIL contention; CPU-bound work can't benefit from threads" \
  --outcome "switched to multiprocessing.Pool; pipeline now achieves 4x speedup on 8 cores" \
  --errors "initially tried ThreadPoolExecutor — no speedup, GIL is the bottleneck" \
  --tags "python" --tags "concurrency" \
  --success
```

### Other useful commands
```bash
agenttrace list              # list all stored traces
agenttrace inspect <id>      # view a specific trace in full
agenttrace stats             # show count, storage size, oldest/newest
agenttrace prune --days 30   # delete traces older than 30 days
```

## What to Do If No Traces Are Returned

Proceed with the task as normal. Absence of traces means no similar past work exists in the store yet — it is not an error. Save a trace after you succeed so future runs benefit.

## What to Do If the CLI Fails

If `agenttrace` is not installed or throws an unexpected error, note the error and continue without the context. AgentTrace is a memory aid, not a dependency. Your task can proceed without it.

Install if missing:
```bash
pip install agenttrace
```
