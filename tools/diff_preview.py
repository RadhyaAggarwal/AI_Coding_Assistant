"""Shared unified-diff rendering for confirmation prompts.

Motivated by a live-observed gap, not a hypothetical one: edit_file's
confirmation prompt used to show only the raw {"search": ..., "replace":
...} arguments, with no view of the actual before/after content. Across
a long --continue chain chasing a subtle bug, that made it hard to tell
by eye whether a given edit was progress or a regression -- a human
approving each individual confirmation had no easy way to compare "this
edit" against "the one three turns ago" from a bare argument dump.

Deliberately language-agnostic: a diff compares lines of text, nothing
more -- unlike tools/syntax_check.py, it needs no per-language grammar,
so it already applies identically to a Python, JS, CSS, or HTML edit
with no extra work.
"""
import difflib
import sys

_MAX_DIFF_LINES = 40
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_CYAN = "\x1b[36m"
_RESET = "\x1b[0m"


def unified_diff_preview(old_content: str, new_content: str, path: str) -> str:
    """A capped unified diff between old_content and new_content, or a
    plain note if there's no textual difference (shouldn't normally
    happen for a real edit, but content-identical replace values are
    possible) or the diff is fully truncated.
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(old_lines, new_lines, fromfile=f"{path} (before)", tofile=f"{path} (after)")
    )
    if not diff_lines:
        return "(no textual difference)"
    if len(diff_lines) > _MAX_DIFF_LINES:
        omitted = len(diff_lines) - _MAX_DIFF_LINES
        diff_lines = diff_lines[:_MAX_DIFF_LINES] + [f"... ({omitted} more diff line(s) omitted)\n"]
    return "".join(diff_lines)


def colorize_diff(diff_text: str) -> str:
    """Wraps added/removed lines of a unified_diff_preview() result in
    ANSI color (green/red, the familiar `git diff` convention) and hunk
    headers in cyan, for a human reading the confirmation prompt in a
    real terminal -- a plain +/- unified diff works but is genuinely
    harder to scan at a glance than a colored one.

    Skipped automatically when stdout isn't a terminal (piped/redirected
    output, or a future non-interactive frontend) so raw escape codes
    never leak somewhere they won't be interpreted -- the same auto-
    detection git itself uses, rather than always-on or a config flag
    nobody would remember to set.
    """
    if not sys.stdout.isatty():
        return diff_text
    colored_lines = []
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("+++") or line.startswith("---"):
            colored_lines.append(line)  # file header, not a content line
        elif line.startswith("+"):
            colored_lines.append(f"{_GREEN}{line}{_RESET}")
        elif line.startswith("-"):
            colored_lines.append(f"{_RED}{line}{_RESET}")
        elif line.startswith("@@"):
            colored_lines.append(f"{_CYAN}{line}{_RESET}")
        else:
            colored_lines.append(line)
    return "".join(colored_lines)


def content_preview(content: str, max_lines: int = 20) -> str:
    """A capped preview of whole-file content, for a genuinely new file
    with nothing to diff against."""
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    omitted = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n... ({omitted} more line(s) omitted)"
