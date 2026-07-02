"""Search/replace file-editing tool — the only way agent logic writes to
a file (CLAUDE.md: "Never write directly to a file from agent logic. All
edits go through the patch / search-replace system, which validates
before applying"). ToolRegistry snapshots the target path before run()
executes (see Tool.target_path() and state/snapshot.py), so any edit
this tool makes can be rolled back afterward.

Two modes:
  - Create a new file: the target must not already exist, and 'search'
    must be empty.
  - Edit an existing file: 'search' must appear in the file's current
    content exactly once. Zero matches means nothing to replace (a wrong
    assumption about the file's content); more than one is ambiguous
    about which occurrence was meant. Either is refused rather than
    guessed at — the same safe pattern most search/replace edit tools
    use, and what CLAUDE.md's "validates before applying" rule is
    pointing at.

Like run_command, this requires human confirmation before it runs —
unlike read-only tools, its effect isn't fully contained by path
validation alone.
"""
from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.path_safety import resolve_within_root


class EditFileError(Exception):
    """Raised when a proposed edit can't be safely/unambiguously applied."""


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Create a new file or make a search/replace edit to an existing "
        "file inside the project. To create a new file, leave 'search' "
        "empty and put the full content in 'replace'. To edit an "
        "existing file, 'search' must be an exact substring that occurs "
        "exactly once in the file; it will be replaced with 'replace'. "
        "Requires human confirmation before it runs."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path, relative to the project root.",
            },
            "search": {
                "type": "string",
                "description": (
                    "Exact text to find and replace. Empty string means "
                    "'create this file' (the file must not already exist)."
                ),
            },
            "replace": {
                "type": "string",
                "description": "Replacement text (or full content, when creating a file).",
            },
        },
        "required": ["path", "search", "replace"],
    }
    requires_confirmation = True

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def target_path(self, arguments: dict[str, Any]) -> Path:
        return resolve_within_root(self._project_root, arguments["path"])

    def run(self, path: str, search: str, replace: str) -> str:
        resolved = resolve_within_root(self._project_root, path)

        if not resolved.is_file():
            if search != "":
                raise EditFileError(
                    f"'{path}' doesn't exist yet — to create it, call again with search=''."
                )
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(replace, encoding="utf-8")
            return f"Created {path}."

        current = resolved.read_text(encoding="utf-8")
        if search == "":
            raise EditFileError(f"'{path}' already exists — provide a non-empty 'search' to edit it.")

        occurrences = current.count(search)
        if occurrences == 0:
            raise EditFileError(
                f"'search' text was not found in '{path}'. No changes made. "
                "'search' must be an exact literal substring of the file's "
                "current content (not a regex or pattern) — call read_file "
                "on it first if you don't already know its exact text."
            )
        if occurrences > 1:
            raise EditFileError(
                f"'search' text appears {occurrences} times in '{path}' — ambiguous. "
                "Include more surrounding context so it matches exactly once."
            )

        resolved.write_text(current.replace(search, replace), encoding="utf-8")
        return f"Edited {path}."
