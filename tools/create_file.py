"""Whole-file write tool: creates a new file, or replaces an existing
file's ENTIRE content. Split out from edit_file (which only ever makes a
targeted change to an existing file's exact matched substring) because a
model repeatedly reached for edit_file's old "empty search means create"
mode against files that already existed, discarding the rest of the
file's content — observed live three times, twice after edit_file's own
description was made more explicit that this was wrong. Naming this as
its own tool means choosing to replace a whole file is an explicit tool
choice by name, not a subtle argument value inside a tool that's
otherwise incapable of touching anything outside an exact match.

Like edit_file/run_command, this requires human confirmation before it
runs. Unlike them, the raw arguments alone don't make the consequences
obvious when the target already exists — confirmation_message() warns
explicitly that this discards the file's current content, rather than
just echoing the call.

Content is also checked with tools/syntax_check.py before being written
(see that module for the live case that motivated it) — a whole-file
write is exactly the shape most likely to embed a large, quote-heavy
blob of source into a single JSON argument, so it's at least as exposed
to the same escaping mistake as edit_file's smaller replacements.
"""
from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.path_safety import resolve_within_root
from tools.syntax_check import InvalidSyntaxError, check_syntax


class CreateFileError(Exception):
    """Raised when content can't be safely written -- e.g. it would leave
    the file syntactically invalid in a language this project can check."""


class CreateFileTool(Tool):
    name = "create_file"
    description = (
        "Create a new file, or replace an existing file's ENTIRE "
        "content. Use this for a brand-new file, or when you genuinely "
        "want to rewrite a file from scratch. For a targeted change to "
        "part of an existing file, use edit_file instead — that only "
        "touches the exact text matched and leaves the rest of the file "
        "alone, while this tool discards everything else the file "
        "currently contains. Requires human confirmation before it runs."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path, relative to the project root.",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write.",
            },
        },
        "required": ["path", "content"],
    }
    requires_confirmation = True

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def target_path(self, arguments: dict[str, Any]) -> Path:
        return resolve_within_root(self._project_root, arguments["path"])

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return f"Writing {arguments['path']}..."

    def confirmation_message(self, arguments: dict[str, Any]) -> str:
        path = arguments["path"]
        resolved = resolve_within_root(self._project_root, path)
        if resolved.is_file():
            try:
                current_lines = len(resolved.read_text(encoding="utf-8").splitlines())
                new_lines = len(arguments["content"].splitlines())
                return (
                    f"Agent wants to run 'create_file' on '{path}', which "
                    f"already exists ({current_lines} lines) — this will "
                    f"REPLACE its ENTIRE content with {new_lines} new "
                    "lines, discarding everything else currently in the "
                    "file."
                )
            except (OSError, UnicodeDecodeError):
                pass  # fall through to the generic message below
        return f"Agent wants to run 'create_file' with arguments {arguments}"

    def run(self, path: str, content: str) -> str:
        resolved = resolve_within_root(self._project_root, path)

        try:
            check_syntax(resolved, content)
        except InvalidSyntaxError as exc:
            raise CreateFileError(str(exc))

        existed = resolved.is_file()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"{'Replaced' if existed else 'Created'} {path}."
