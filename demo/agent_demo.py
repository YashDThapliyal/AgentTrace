"""
AgentTrace demo — real agent loop using OpenAI gpt-4o-mini.

Shows the core AgentTrace value prop across 4 rounds:
  Round 1: cold start (no context)
  Round 2: similar task → gets Round 1 context
  Round 3: unrelated task → cold start again
  Round 4: similar to Round 3 → gets Round 3 context

Run:
  pip install agenttrace openai
  python demo/agent_demo.py
"""
from __future__ import annotations

import os
import sys

from openai import OpenAI

import agenttrace

# ─── Validate API key early ───────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print(
        "Error: OPENAI_API_KEY not set in environment. "
        "Add it to ~/.zshrc and run: source ~/.zshrc"
    )
    sys.exit(1)

# ─── Initialize ───────────────────────────────────────────────────────────────
# Clear demo store before each run for clean results.
if os.path.exists("./demo/demo_traces.jsonl"):
    os.remove("./demo/demo_traces.jsonl")

# Separate store so this demo doesn't touch your real trace library.
agenttrace.init(store_path="./demo/demo_traces.jsonl", threshold=0.5)

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"

# Track results for the summary
summary: list[dict] = []


# ─── Agent ────────────────────────────────────────────────────────────────────
def run_agent(task: str) -> tuple[str, str]:
    """Run the agent on a task.

    Returns (recall_context, agent_response).
    """
    context = agenttrace.recall(task)

    system_prompt = "You are a helpful coding assistant. Be concise and practical."
    if context:
        system_prompt += (
            "\n\nYou have access to the following reasoning traces from similar "
            "past tasks. Use them to inform your approach.\n\n" + context
        )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ],
        max_tokens=400,
    )

    return context, response.choices[0].message.content or ""


def print_round(
    number: int,
    task: str,
    context: str,
    response: str,
    *,
    saved: bool = True,
) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  ROUND {number}")
    print(f"{'═' * width}")
    print(f"  Task: {task}")
    print()

    if context:
        # Extract similarity scores from the context block for display
        scores = [
            line.split("similarity: ")[1].split(" ")[0]
            for line in context.splitlines()
            if "similarity:" in line
        ]
        print(f"  ◆ Recall: {len(scores)} trace(s) injected — scores: {', '.join(scores)}")
        print()
        print("  --- Context block ---")
        for line in context.splitlines():
            print(f"  {line}")
        print()
    else:
        print("  ◇ Recall: no matching traces (cold start)")
        print()

    print("  --- Agent response ---")
    for line in response.splitlines():
        print(f"  {line}")
    if saved:
        print()
        print("  ✓ Trace saved")


# ─── Round 1: cold start ──────────────────────────────────────────────────────
task1 = "Write a Python function that finds all duplicate values in a list"
context1, response1 = run_agent(task1)

print_round(1, task1, context1, response1)

trace_id1 = agenttrace.save(
    task=task1,
    reasoning=(
        "Used a dictionary to count occurrences of each element. "
        "Collected elements with count > 1 into a set to avoid duplicates in output."
    ),
    outcome=response1,
    errors=[],
    success=True,
    model=MODEL,
    tags=["python", "lists", "deduplication"],
)
summary.append({
    "round": 1,
    "task": task1,
    "got_context": bool(context1),
    "context_scores": [],
    "trace_id": trace_id1,
})


# ─── Round 2: similar task — should surface Round 1 ──────────────────────────
task2 = "Write a Python function that removes duplicate entries from a list"
context2, response2 = run_agent(task2)

print_round(2, task2, context2, response2)

scores2 = [
    float(line.split("similarity: ")[1].split(" ")[0])
    for line in context2.splitlines()
    if "similarity:" in line
]
trace_id2 = agenttrace.save(
    task=task2,
    reasoning=(
        "Leveraged dict.fromkeys() to preserve order while removing duplicates, "
        "or used a set for order-independent deduplication."
    ),
    outcome=response2,
    errors=[],
    success=True,
    model=MODEL,
    tags=["python", "lists", "deduplication"],
)
summary.append({
    "round": 2,
    "task": task2,
    "got_context": bool(context2),
    "context_scores": scores2,
    "trace_id": trace_id2,
})


# ─── Round 3: unrelated domain — should be cold start ────────────────────────
task3 = "Write a SQL query to calculate monthly revenue by product"
context3, response3 = run_agent(task3)

print_round(3, task3, context3, response3)

scores3 = [
    float(line.split("similarity: ")[1].split(" ")[0])
    for line in context3.splitlines()
    if "similarity:" in line
]
trace_id3 = agenttrace.save(
    task=task3,
    reasoning=(
        "Used DATE_TRUNC or strftime to group by month, joined with product table, "
        "and aggregated revenue with SUM."
    ),
    outcome=response3,
    errors=[],
    success=True,
    model=MODEL,
    tags=["sql", "analytics", "aggregation"],
)
summary.append({
    "round": 3,
    "task": task3,
    "got_context": bool(context3),
    "context_scores": scores3,
    "trace_id": trace_id3,
})


# ─── Round 4: similar to Round 3 — should surface it ─────────────────────────
task4 = "Write a SQL query to find total sales per customer"
context4, response4 = run_agent(task4)

print_round(4, task4, context4, response4, saved=False)

scores4 = [
    float(line.split("similarity: ")[1].split(" ")[0])
    for line in context4.splitlines()
    if "similarity:" in line
]
summary.append({
    "round": 4,
    "task": task4,
    "got_context": bool(context4),
    "context_scores": scores4,
    "trace_id": None,
})


# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'═' * 60}")
print("  SUMMARY")
print(f"{'═' * 60}")
print("  Traces saved: 3  (rounds 1–3; round 4 not saved)\n")

for entry in summary:
    n = entry["round"]
    if entry["got_context"]:
        scores_str = ", ".join(f"{s:.2f}" for s in entry["context_scores"])
        print(f"  Round {n}: context injected  (similarity scores: {scores_str})")
    else:
        print(f"  Round {n}: cold start        (no matching traces)")

print()
print("  Expected pattern:")
print("    Round 1 → cold    (empty store)")
print("    Round 2 → context (duplicate-list traces are semantically similar)")
print("    Round 3 → cold    (SQL domain, nothing in store yet)")
print("    Round 4 → context (SQL aggregation traces are semantically similar)")
print()
print("  Demo store: ./demo/demo_traces.jsonl")
print(f"{'═' * 60}\n")
