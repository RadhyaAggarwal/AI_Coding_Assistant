# Local AI Coding Agent — Project Memory

## What this is
A Python application that acts as an agentic coding assistant, powered by a
locally-hosted open-source LLM (served via Ollama) instead of any third-party
API. It must work across languages (Python, HTML, CSS, JS, etc.), not just
Python. See `Build_Plan.pdf` and `Build_Plan_Addendum.pdf` in the project root
for full background — read both before starting significant work.

## Architecture — module boundaries (do not violate)
- `agent_controller/` — orchestration, planning, the ReAct loop. Owns the
  understand -> plan -> act -> observe -> continue cycle.
- `model_interface/` — the ONLY place that talks to the LLM. Must stay
  model-agnostic: no model-specific logic leaks into agent_controller or tools.
- `tools/` — file, code-search, execution, and git tools. The agent never
  touches the filesystem or shell directly; it always goes through a tool.
- `repo_index/` — tree-sitter based scanning, AST indexing, retrieval.
- `state/` — snapshots, rollback, change tracking. Runs before any edit.

## Rules
- Never write directly to a file from agent logic. All edits go through the
  patch / search-replace system, which validates before applying.
- Model name, endpoint URL, and all local paths live in `config.yaml` —
  never hardcoded into application code.
- Every new tool is registered in a central tool registry with an explicit
  JSON schema (name, description, parameters).
- Use tree-sitter for any non-Python language parsing; use Python's built-in
  `ast` module only for Python-specific analysis.
- Prefer small, reviewable diffs. Do not rewrite entire files when a
  search/replace patch will do.

## Division of labor (important — read this)
Claude Code owns the harness: tool implementations, the Model Interface
abstraction, the repo indexer, the patch engine, state/rollback, test
scaffolding.

Claude Code may also write the FIRST DRAFT of the system prompt and ReAct
loop structure under `agent_controller/prompts/`, based on established
agentic prompting patterns. After that first draft has been tested against
the actual running local model, ownership of further iteration moves to the
human developer — Claude Code should not silently rewrite files under
`agent_controller/prompts/` once they've been marked "tuned" (see a comment
at the top of the file), since changes there should be driven by observed
model behavior, not general knowledge. If asked to revisit a tuned prompt,
propose changes rather than overwriting directly.

## Commands
- Run app: `python main.py`
- Run tests: `pytest tests/`
- Start local model: `ollama run <model-name-from-config.yaml>`

## Current status
Project is in early scaffolding (Phase 0 / minimal vertical slice). Prioritize
a thin end-to-end path (talk to model -> one working tool -> minimal planner)
over building any single subsystem out fully in isolation.
