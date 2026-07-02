# Local AI Coding Agent

A Python agentic coding assistant powered by a locally-hosted open-source
LLM (served via [Ollama](https://ollama.com)) instead of a third-party
API. See `CLAUDE.md` for architecture and module boundaries, and
`AI-Coding Agent.pdf` / `Build_Plan_Addendum.pdf` for the full build plan.

This is currently a **minimal vertical slice**: talk to a local model,
call one tool (`read_file`), return an answer. Later phases (repo
indexing, more tools, safe editing, rollback, etc.) are not built yet.

## Quickstart

1. **Install Ollama** and make sure it's running locally:
   https://ollama.com/download

2. **Pull the model** referenced in `config.yaml` (`model.name`):

   ```
   ollama pull qwen2.5-coder:14b
   ```

   If you want a lighter model for a laptop, pull a smaller one (e.g.
   `qwen2.5-coder:7b`) and update `model.name` in `config.yaml` to match.

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

## Configuration

All model/endpoint/path settings live in `config.yaml` — nothing is
hardcoded in application code. Edit that file to point at a different
Ollama endpoint, model name, or project root.

## Tests

```
pytest tests/
```
