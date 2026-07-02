"""Plain-text search tool ("find X across the project"). Not AST-aware —
that's repo_index/'s job once it's built — this is a simple, scoped,
path-validated substring grep across project files.
"""
from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.path_safety import resolve_within_root

_SKIP_DIR_NAMES = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache"}
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

    def run(self, query: str, path: str = ".") -> str:
        search_root = resolve_within_root(self._project_root, path)
        if not search_root.is_dir():
            raise NotADirectoryError(f"No such directory: {path}")

        needle = query.lower()
        matches: list[str] = []
        for file_path in sorted(search_root.rglob("*")):
            if len(matches) >= _MAX_MATCHES:
                break
            if not file_path.is_file() or any(
                part in _SKIP_DIR_NAMES for part in file_path.parts
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
