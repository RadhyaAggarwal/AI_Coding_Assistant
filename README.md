# Local AI Coding Agent

A Python agentic coding assistant powered by a locally-hosted open-source
LLM (served via [Ollama](https://ollama.com)) instead of a third-party
API. See `CLAUDE.md` for architecture and module boundaries, and
`AI-Coding Agent.pdf` / `Build_Plan_Addendum.pdf` for the full build plan.

The agent understands a project's structure (Python/JavaScript/CSS/HTML
symbol indexing via `ast` and tree-sitter), can search and read files,
run shell commands, and make validated, snapshotted, human-confirmed
edits — chaining multiple tool calls per request (e.g. edit a file, then
run its tests to verify the fix actually works). Not yet built: a
self-correction loop beyond what fits in one request's step budget, and
team/multi-developer features.

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

## Configuration

All model/endpoint/path settings live in `config.yaml` — nothing is
hardcoded in application code. Edit that file to point at a different
Ollama endpoint, model name, project root, or to adjust timeouts.

## Tests

```
pytest tests/
```
