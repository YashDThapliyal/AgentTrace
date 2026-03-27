"""Basic Python API usage example for AgentTrace.

Run this script after `pip install agenttrace` to see the full recall/save cycle.
"""
import agenttrace

# Optional: configure explicitly (defaults work fine too)
agenttrace.init(
    backend="jsonl",
    store_path="~/.agenttrace/examples.jsonl",
    top_k=3,
    threshold=0.75,
)

# -----------------------------------------------------------------------
# 1. Before starting a task — recall similar past traces
# -----------------------------------------------------------------------
task = "fix the threading bug in pipeline.py"
context = agenttrace.recall(task)

if context:
    print("=== Past context ===")
    print(context)
    print()
else:
    print("No past traces found. Starting fresh.\n")

# -----------------------------------------------------------------------
# 2. Agent does its work here ...
# (In a real scenario this is where your agent runs)
# -----------------------------------------------------------------------

print("Working on task...")
reasoning = (
    "Identified GIL contention — threads cannot parallelize CPU-bound work in Python. "
    "Replaced ThreadPoolExecutor with multiprocessing.Pool using the same worker function."
)
outcome = "Pipeline now achieves 4x speedup on 8 cores."

# -----------------------------------------------------------------------
# 3. After success — save the trace
# -----------------------------------------------------------------------
trace_id = agenttrace.save(
    task=task,
    reasoning=reasoning,
    outcome=outcome,
    errors=[
        "Initially tried thread pool executor — same performance, no improvement.",
        "Tried increasing thread count to 16 — still no speedup due to GIL.",
    ],
    success=True,
    model="claude-sonnet-4-6",
    tags=["python", "concurrency", "performance"],
)
print(f"Trace saved: {trace_id}")

# -----------------------------------------------------------------------
# 4. Next time — recall will surface this trace
# -----------------------------------------------------------------------
print("\n=== Recalling now ===")
context = agenttrace.recall("python multiprocessing slow compared to expected")
print(context if context else "No matching traces found.")
