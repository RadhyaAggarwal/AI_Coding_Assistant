"""Shared source-snippet helper for tools that report a file:line
location. Every such tool should show real content there rather than let
the model guess/fabricate what's at that location — a recurring failure
mode found in find_symbol earlier (a bare file:line location led the
model to invent plausible-looking content to fill the gap; one source
line wasn't even enough — a CSS rule's opening brace alone looked
"empty" and the model confidently reported wrong properties).
"""
from pathlib import Path

_SNIPPET_LINES = 6


def snippet_at(project_root: Path, relative_path: str, line: int) -> str:
    try:
        lines = (project_root / relative_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return "    (source unavailable)"

    start = line - 1
    if start < 0 or start >= len(lines):
        return "    (source unavailable)"

    window = lines[start : start + _SNIPPET_LINES]
    rendered = "\n".join(f"    {text}" for text in window)
    if start + _SNIPPET_LINES < len(lines):
        rendered += "\n    ..."
    return rendered
