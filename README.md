# Local AI Coding Agent

A Python agentic coding assistant powered by a locally-hosted open-source
LLM (served via [Ollama](https://ollama.com)) instead of a third-party
API. See `CLAUDE.md` for architecture and module boundaries, and
`AI-Coding Agent.pdf` / `Build_Plan_Addendum.pdf` for the full build plan.

The agent understands a project's structure (Python/JavaScript/CSS/HTML
symbol indexing via `ast` and tree-sitter, cached to disk), can search
and read files, run shell commands, and make validated, snapshotted,
human-confirmed changes — chaining multiple tool calls per request (e.g.
edit a file, then run its tests to verify the fix actually works). File
changes are split into two explicit tools: `edit_file` makes a targeted
search/replace change to part of an existing file, and `create_file`
creates a new file or deliberately replaces one's entire content — kept
separate so "replace everything" is always an explicit tool choice, not
a subtle argument value inside an edit that's otherwise incapable of
touching more than the exact text matched. Beyond file/symbol lookup
(`find_symbol`, `repo_overview`, `html_overview`), it can also trace
cross-file relationships — which files import a given module
(`find_importers`) and where a function is actually called from
(`find_callers`) — for questions that span more than one file.

Optionally, if a separate embedding model is configured (see
"Semantic search" below), the agent also gets `semantic_search`: finds
relevant code for a vague, natural-language question when you don't
know a function's exact name, ranking by meaning rather than literal
text match. This is a genuinely different capability from the
exact-match tools above, not a replacement for them — see that section
for an honest note on how reliably the agent actually chooses to use it.

Every request is logged (`--sessions` to browse recent ones), and a
single task can be continued across separate `python main.py` calls
with `--continue` if it runs out of its step budget partway through.

Not yet built: a dedicated Git-specific tool (`run_command` can already
run raw git commands, but there's no structured tool for it) and a
dedicated testing/verification subsystem beyond what the existing loop
already does (running tests via `run_command` and checking the result).
Team/multi-developer *awareness* features (shared history, cross-user
context) also aren't built — sharing a single model server across a
team, which is a different and already-supported thing, is covered in
`DEPLOYMENT.md`.

## Quickstart

1. **Install Ollama** and make sure it's running locally:
   https://ollama.com/download

2. **Pull the model** referenced in `config.yaml` (`model.name`):

   ```
   ollama pull qwen2.5-coder:7b
   ```

   The default targets a modest machine (16GB RAM, no dedicated GPU). If
   you have a stronger GPU, a larger model (e.g. `qwen2.5-coder:14b`)
   will be noticeably more reliable at multi-step tool use — pull it and
   update `model.name` in `config.yaml` to match.

3. **Install Python dependencies** (Python 3.10+):

   ```
   pip install -r requirements.txt
   ```

4. **Run the agent**:

   ```
   python main.py "Summarize config.yaml"
   ```

   or run it with no arguments to be prompted interactively:

   ```
   python main.py
   ```

   Some tool calls (running a shell command, editing a file) will ask
   for your confirmation (`Allow? [y/N]`) before they execute. On
   CPU-only inference, expect each model call to take anywhere from
   ~1 to a few minutes, especially for requests that chain several tool
   calls.

5. **Review or undo an edit** the agent made:

   ```
   python main.py --history
   python main.py --rollback <snapshot_id>
   ```

6. **Continue a task, or look back at past ones**:

   ```
   python main.py --continue "keep going"   # resume the last task with a fresh step budget
   python main.py --sessions                # list recent requests and their outcomes
   ```

## Semantic search (optional)

`semantic_search` finds code by meaning rather than exact name or text
match — useful for a vague question like "where do we check if a user
is allowed to delete something" when you don't know what the relevant
function is actually called. It's disabled by default and requires a
separate, much smaller embedding model:

```
ollama pull nomic-embed-text
```

Then set `model.embedding_name: "nomic-embed-text"` in `config.yaml`
(commented out by default). With nothing set, the tool simply isn't
registered — no error, no degraded behavior for anyone not using it.

**Honest note on real-world usefulness, not just whether it works:**
the underlying search itself (chunking, ranking) is tested and correct.
Live testing found the coding model in this project rarely chooses to
use it on its own, even when it's clearly available and highly ranked —
it tends to reach for an exact-match tool first and give up rather than
try the meaning-based one. This is a model-judgment limitation, not a
bug in the tool, and it may well improve with a more capable coding
model. Kept enabled (opt-in, zero cost if unconfigured) rather than
removed over a limitation outside the tool's own control — see project
history for the full investigation if picking this up.

There's also an opt-in, off-by-default `agent.embedding_aware_routing`
config flag that makes per-step tool selection consider embedding
similarity alongside keyword overlap, not just for `semantic_search`
itself — see the comment above it in `config.yaml` for what it does and
why it defaults to off (a real added network call on every step, not
just when a tool is actually used).

## Configuration

All model/endpoint/path settings live in `config.yaml` — nothing is
hardcoded in application code. Edit that file to point at a different
Ollama endpoint, model name, project root, or to adjust timeouts.

## Sharing one model server across a team

If several people want to share a single Ollama server on the same
network instead of each installing Ollama and downloading the model
individually, see `DEPLOYMENT.md`.

## Tests

```
pytest tests/
```
