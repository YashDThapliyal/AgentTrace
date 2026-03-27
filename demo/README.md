# AgentTrace Demo

A live demo of AgentTrace in a real agent loop using OpenAI gpt-4o-mini.

## What it shows

Four agent rounds that demonstrate the core AgentTrace value prop:

| Round | Task | Expected |
|---|---|---|
| 1 | Find duplicate values in a list | Cold start — no context yet |
| 2 | Remove duplicate entries from a list | Gets Round 1 context (semantically similar) |
| 3 | SQL query: monthly revenue by product | Cold start — new domain |
| 4 | SQL query: total sales per customer | Gets Round 3 context (semantically similar) |

The demo uses a separate store (`demo/demo_traces.jsonl`) so it doesn't touch your main trace library.

## Setup

```bash
pip install agenttrace openai
```

Make sure `OPENAI_API_KEY` is set:

```bash
# Already in ~/.zshrc — just make sure it's sourced:
source ~/.zshrc
echo $OPENAI_API_KEY   # should print your key
```

## Run

From the project root:

```bash
python demo/agent_demo.py
```

## Reset

To run the demo fresh (clear saved traces):

```bash
rm demo/demo_traces.jsonl
python demo/agent_demo.py
```
