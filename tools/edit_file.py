"""Search/replace file-editing tool — the only way agent logic makes a
targeted change to an existing file (CLAUDE.md: "Never write directly to
a file from agent logic. All edits go through the patch / search-replace
system, which validates before applying"). ToolRegistry snapshots the
target path before run() executes (see Tool.target_path() and
state/snapshot.py), so any edit this tool makes can be rolled back
afterward.

Only edits an existing file: 'search' must be a non-empty exact
substring of its current content, occurring exactly once. Zero matches
means nothing to replace (a wrong assumption about the file's content);
more than one is ambiguous about which occurrence was meant; empty is
refused outright. Either is refused rather than guessed at — the same
safe pattern most search/replace edit tools use, and what CLAUDE.md's
"validates before applying" rule is pointing at.

This tool used to also create new files (an empty 'search' meant
"create"), but a model reached for that mode against files that already
existed — writing only a partial replacement as 'replace' and silently
discarding the rest of the file's content — three times live, twice
*after* this tool's own description was made more explicit that empty
search was create-only. See tools/create_file.py, which now owns whole-
file creation/replacement as a separate, explicitly-named tool: choosing
to replace an entire file is now a distinct tool choice, not a subtle
argument value inside a tool that's otherwise incapable of discarding
content outside the exact substring it matched.

Like run_command, this requires human confirmation before it runs —
unlike read-only tools, its effect isn't fully contained by path
validation alone.

The resulting content is also checked with tools/syntax_check.py before
being written (see that module for the live case that motivated it: a
model double-escaping a docstring's quotes, valid JSON either way, but
invalid Python once decoded). A bad edit is refused with the real parse
error instead of corrupting the file.
"""
from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.diff_preview import colorize_diff, unified_diff_preview
from tools.path_safety import resolve_within_root
from tools.syntax_check import InvalidSyntaxError, check_syntax


class EditFileError(Exception):
    """Raised when a proposed edit can't be safely/unambiguously applied."""


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Make a targeted search/replace edit to an EXISTING file. "
        "'search' must be a non-empty, exact substring of the file's "
        "current content that occurs exactly once; it will be replaced "
        "with 'replace'. If you don't already know the file's exact "
        "current content, call read_file on it first — guessing at "
        "'search' is refused, not guessed at. To create a new file, or "
        "to replace an existing file's entire content, use create_file "
        "instead — this tool never does that. Requires human "
        "confirmation before it runs."
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
                    "Exact, non-empty text to find and replace — a literal "
                    "substring of the file's current content."
                ),
            },
            "replace": {
                "type": "string",
                "description": "Replacement text for the matched 'search' substring.",
            },
        },
        "required": ["path", "search", "replace"],
    }
    requires_confirmation = True

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def target_path(self, arguments: dict[str, Any]) -> Path:
        return resolve_within_root(self._project_root, arguments["path"])

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return f"Editing {arguments['path']}..."

    def confirmation_message(self, arguments: dict[str, Any]) -> str:
        """Shows a real before/after diff when it's safe to compute one,
        rather than just the raw {"search": ..., "replace": ...}
        arguments -- live-observed gap: across a long --continue chain
        chasing a bug, a bare argument dump made it hard to tell by eye
        whether a given edit was progress or a regression relative to a
        few turns earlier. Falls back to the generic message whenever the
        diff can't be safely computed here (missing arguments, the file
        doesn't exist, or 'search' doesn't match exactly once) -- run()
        does the real validation and will raise a specific error either
        way; this is purely a preview, not a second source of truth.
        """
        path = arguments.get("path")
        search = arguments.get("search")
        replace = arguments.get("replace")
        if not path or not search:
            return super().confirmation_message(arguments)
        try:
            resolved = resolve_within_root(self._project_root, path)
            current = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return super().confirmation_message(arguments)
        if current.count(search) != 1:
            return super().confirmation_message(arguments)
        new_content = current.replace(search, replace if replace is not None else "")
        diff = unified_diff_preview(current, new_content, path)
        return f"Agent wants to edit '{path}':\n{colorize_diff(diff)}"

    def run(self, path: str, search: str, replace: str) -> str:
        resolved = resolve_within_root(self._project_root, path)

        if not resolved.is_file():
            raise EditFileError(
                f"'{path}' doesn't exist — edit_file only makes targeted "
                "changes to existing files. To create it, call create_file "
                "instead."
            )

        if search == "":
            raise EditFileError(
                "'search' must not be empty — edit_file only makes targeted "
                "changes to existing content and never discards the rest of "
                "a file. To replace this file's entire content, call "
                "create_file instead."
            )

        current = resolved.read_text(encoding="utf-8")
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

        new_content = current.replace(search, replace)
        try:
            check_syntax(resolved, new_content)
        except InvalidSyntaxError as exc:
            raise EditFileError(str(exc))

        resolved.write_text(new_content, encoding="utf-8")
        return f"Edited {path}."
