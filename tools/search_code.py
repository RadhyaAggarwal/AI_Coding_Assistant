"""Plain-text search tool ("find X across the project"). Not AST-aware —
that's repo_index/'s job once it's built — this is a simple, scoped,
path-validated substring grep across project files.

Reuses repo_index.scanner's skip-dir list rather than keeping its own
copy -- this tool used to maintain a separate, near-identical set that
was missing ".agent_state" (repo_index's list already excludes it).
Live-observed real consequence, not theoretical: once --continue and the
session log started writing conversation history into .agent_state/,
search_code results got buried under noise from the agent's own past
conversation turns (which themselves quote earlier search results,
compounding across successive --continue calls) -- a real match
(scratch_receipt.py's generate_receipt_total) was present in the output
the whole time but the model never acted on it, surrounded by 16 lines
of self-referential conversation-log noise. A single shared list is the
actual fix, not just adding ".agent_state" here too, since two
independently-maintained copies is exactly how they drifted apart in the
first place.
"""
from pathlib import Path
from typing import Any

from repo_index.scanner import SKIP_DIR_NAMES
from tools.base import Tool
from tools.path_safety import resolve_within_root
from tools.path_suggestions import suggest_similar_directories

_MAX_MATCHES = 50


class SearchCodeTool(Tool):
    name = "search_code"
    description = (
        "Search for a text string across files in the project (a plain, "
        "case-insensitive substring search — not a regex). Returns "
        "matching 'path:line: content' entries, capped at 50 matches. "
        "Optionally scope the search to a subdirectory."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search for.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Directory to search within, relative to the project "
                    "root. Defaults to the whole project."
                ),
            },
        },
        "required": ["query"],
    }

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return f"Searching for '{arguments['query']}'..."

    def run(self, query: str, path: str = ".") -> str:
        search_root = resolve_within_root(self._project_root, path)
        if not search_root.is_dir():
            message = f"No such directory: {path}"
            suggestions = suggest_similar_directories(self._project_root, path)
            if suggestions:
                message += ". Did you mean one of these real directories? " + ", ".join(suggestions)
            raise NotADirectoryError(message)

        needle = query.lower()
        matches: list[str] = []
        for file_path in sorted(search_root.rglob("*")):
            if len(matches) >= _MAX_MATCHES:
                break
            if not file_path.is_file() or any(
                part in SKIP_DIR_NAMES for part in file_path.parts
            ):
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            relative = file_path.relative_to(self._project_root)
            for line_number, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    matches.append(f"{relative}:{line_number}: {line.strip()}")
                    if len(matches) >= _MAX_MATCHES:
                        break

        return "\n".join(matches) if matches else "No matches found."
