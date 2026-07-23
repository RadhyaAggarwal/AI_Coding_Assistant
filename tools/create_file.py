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
from tools.diff_preview import colorize_diff, content_preview, unified_diff_preview
from tools.path_safety import resolve_within_root
from tools.syntax_check import NOTE_MARKER, InvalidSyntaxError, advisory_notes, check_syntax


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
    # See Tool.always_mutates -- this tool never returns without having
    # actually written the file (every failure case raises instead).
    always_mutates = True

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def should_show_result(self, result: str) -> bool:
        # See Tool.should_show_result -- an ordinary create/replace stays
        # silent (unchanged from before), but a note is worth surfacing to
        # the human live, not just left for the model to maybe mention.
        return NOTE_MARKER in result

    def target_path(self, arguments: dict[str, Any]) -> Path:
        return resolve_within_root(self._project_root, arguments["path"])

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return f"Writing {arguments['path']}..."

    def confirmation_message(self, arguments: dict[str, Any]) -> str:
        path = arguments["path"]
        content = arguments.get("content", "")
        resolved = resolve_within_root(self._project_root, path)
        if resolved.is_file():
            try:
                current = resolved.read_text(encoding="utf-8")
                new_lines = len(content.splitlines())
                diff = unified_diff_preview(current, content, path)
                return (
                    f"Agent wants to run 'create_file' on '{path}', which "
                    f"already exists ({len(current.splitlines())} lines) — "
                    f"this will REPLACE its ENTIRE content with {new_lines} "
                    f"new lines, discarding everything else currently in "
                    f"the file:\n{colorize_diff(diff)}"
                )
            except (OSError, UnicodeDecodeError):
                pass  # fall through to the generic message below
        # Genuinely new file -- nothing to diff against, so show a
        # preview of what's about to be written instead.
        return f"Agent wants to create '{path}' ({len(content.splitlines())} lines):\n{content_preview(content)}"

    def run(self, path: str, content: str) -> str:
        resolved = resolve_within_root(self._project_root, path)

        try:
            check_syntax(resolved, content)
        except InvalidSyntaxError as exc:
            raise CreateFileError(str(exc))

        existed = resolved.is_file()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        result = f"{'Replaced' if existed else 'Created'} {path}."
        notes = advisory_notes(resolved, content)
        if notes:
            return result + NOTE_MARKER + NOTE_MARKER.join(notes)
        return result
