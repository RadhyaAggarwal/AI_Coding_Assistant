"""Derives which files a conversation actually changed, straight from
the transcript's own tool results -- never from the model's own
narration of what it did.

Motivated by a live-observed failure of the alternative (telling the
model itself, via an injected fact, whether it had made a real change):
that approach worked for the case it targeted, but the same loop-exit
path fires for reasons unrelated to needing a fix (e.g. running out of
wasted-step budget on an ordinary info-only query), and the injected
note's fix-specific wording derailed an otherwise-correct answer into
describing "no files were changed" instead of answering the question
that had already been fully answered. Reverted in favor of this: never
feed a fact back into the model's own context where it risks
influencing behavior it was never meant to affect -- just tell the
human directly, the same "ground with real facts, don't trust
narration" principle this project already relies on for diffs
(tools/diff_preview.py) and real command output (Tool.show_result),
just surfaced alongside the answer instead of fed back into the loop.

Deliberately mechanical, not a summary the model writes: edit_file and
create_file only ever return without raising when they've genuinely
written a file (see Tool.always_mutates), and their success messages'
FIRST LINE has a fixed, stable shape ("Edited {path}.", "Created
{path}.", "Replaced {path}.") -- checked directly against no other tool
in this project producing a colliding prefix, so matching it is exact,
not a heuristic guess. Only the first line is guaranteed fixed, though:
an advisory_notes() note, when present, is appended as further lines
after it (see tools/syntax_check.py's NOTE_MARKER) -- live-observed real
consequence of matching the whole string instead of just the first line
once notes existed: a note's own text got swept into what should have
been a clean path, corrupting the printed summary. Fixed by only ever
matching against message.content.split("\n", 1)[0].
"""
from model_interface.base import Message

_EDIT_FILE_PREFIX = "Edited "
_CREATE_FILE_PREFIXES = ("Created ", "Replaced ")


def changed_files(transcript: list[Message]) -> list[str]:
    """Paths of every file a successful edit_file/create_file call
    actually wrote, in first-occurrence order, deduplicated.

    The confirmation ("Edited {path}." / "Created {path}." /
    "Replaced {path}.") is always the first line of the result -- an
    advisory_notes() note, if any, is always appended after it as
    further lines (see tools/syntax_check.py's NOTE_MARKER). Matching
    against the first line only, not the whole content, keeps a note's
    text from being swept into what should be a clean path string.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for message in transcript:
        if message.role != "tool":
            continue
        first_line = message.content.split("\n", 1)[0]
        path = None
        if first_line.startswith(_EDIT_FILE_PREFIX) and first_line.endswith("."):
            path = first_line[len(_EDIT_FILE_PREFIX):-1]
        else:
            for prefix in _CREATE_FILE_PREFIXES:
                if first_line.startswith(prefix) and first_line.endswith("."):
                    path = first_line[len(prefix):-1]
                    break
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths
