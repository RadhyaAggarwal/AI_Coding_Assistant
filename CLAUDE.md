# Local AI Coding Agent — Project Memory

## What this is
A Python application that acts as an agentic coding assistant, powered by a
locally-hosted open-source LLM (served via Ollama) instead of any third-party
API. It must work across languages (Python, HTML, CSS, JS, etc.), not just
Python. See `AI-Coding Agent.pdf` and `Build_Plan_Addendum.pdf` in the project root
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
- Start local model server: `ollama serve` (this project talks to its HTTP
  API, not the `ollama run` chat CLI — see `DEPLOYMENT.md` if that
  distinction is unfamiliar)

## Current status
No longer early scaffolding — see `README.md` for a full capability
overview. In brief: a working multi-step ReAct loop (`agent_controller/`)
with per-step tool routing, duplicate-call detection, and context-window
budgeting; 13 tools including file read/edit/create, shell execution,
AST/tree-sitter-based symbol/relationship lookup (`find_symbol`,
`find_importers`, `find_callers`), and an opt-in embeddings-based
`semantic_search`; snapshot/rollback for every edit; session logging
(`--sessions`) and short-term conversation continuation (`--continue`);
400+ tests. Deliberately not built, left for whoever picks this up next
(chosen as a good fit for someone newer to AI/ML methods specifically,
not an oversight): Git-specific tools (`run_command` can already run raw
git commands, but there's no dedicated, structured tool for it), and a
possible dedicated testing/verification subsystem beyond what the
existing loop + `run_command` + `edit_file` already covers. See project
memory / commit history for the detailed history of what's been tried,
what worked, and what's an accepted model-capability limitation rather
than an open bug.
