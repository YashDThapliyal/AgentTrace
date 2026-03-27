# AgentTrace

## What this project is
A lightweight Python library that saves successful agent reasoning traces and surfaces them on future runs. Read SPEC.md for the full project spec before doing anything.

## Build order
Work in this exact order — each module depends on the one before it:
1. config.py
2. storage/base.py → storage/jsonl.py → storage/sqlite.py
3. embeddings/base.py → embeddings/local.py → embeddings/openai.py → embeddings/anthropic.py
4. retrieval.py
5. injection.py
6. core.py
7. __init__.py
8. cli.py
9. skill/SKILL.md
10. examples/, tests/, README.md, pyproject.toml

## Rules
- Read SPEC.md fully before writing any code
- Build and test each module independently before moving on
- Every module must have a corresponding test file
- Run ruff and mypy before marking any module done
- Do not build anything listed under "Out of Scope for v0.1" in the spec

## Commands
- Run tests: pytest tests/
- Lint: ruff check .
- Type check: mypy agenttrace/
- Install dev deps: pip install -e ".[dev]"
