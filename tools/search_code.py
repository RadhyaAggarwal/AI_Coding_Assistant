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
        "case-insensitive substring search, not a regex). Returns "
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

    def __init__(self, project_root: str | Path, semantic_search_available: bool = False):
        self._project_root = Path(project_root).resolve()
        # Live-observed gap: this tool only finds an exact literal
        # substring, and a model that guesses a plausible-sounding phrase
        # (e.g. "deduplicate tasks") gets "No matches found" and then
        # concludes no such mechanism exists, rather than reaching for
        # semantic_search -- even when it's available and its own
        # description says it's for exactly this case. Nudging here,
        # right where the miss actually happens, mirrors the same
        # give-real-grounding-instead-of-a-gap-to-guess-into pattern
        # already used for edit_file's error message and
        # path_suggestions's "did you mean" hints. Optional and off by
        # default so this tool doesn't reference a tool that might not be
        # registered.
        self._semantic_search_available = semantic_search_available
        if semantic_search_available:
            # Extends the class-level default on this instance only
            # (Tool.schema() reads self.description, which resolves here
            # before falling back to the class attribute) -- proactive
            # counterpart to the runtime nudge above: this fires before a
            # wasted attempt, not just after one fails. Mirrors
            # semantic_search's own description, which already points the
            # other way ("use find_symbol or search_code instead").
            self.description = self.description + (
                " If you don't know the exact text to search for, try "
                "semantic_search instead."
            )

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

        if matches:
            return "\n".join(matches)
        if self._semantic_search_available:
            return (
                "No matches found. This only checks for an exact literal "
                "substring -- if you're not sure of the exact wording or "
                "name to search for (you know what something does, not "
                "what it's called), try semantic_search instead."
            )
        return "No matches found."
