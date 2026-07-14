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
written a file (see Tool.always_mutates), and their success messages
have a fixed, stable shape ("Edited {path}.", "Created {path}.",
"Replaced {path}.") -- checked directly against no other tool in this
project producing a colliding prefix, so matching them is exact, not a
heuristic guess.
"""
from model_interface.base import Message

_EDIT_FILE_PREFIX = "Edited "
_CREATE_FILE_PREFIXES = ("Created ", "Replaced ")


def changed_files(transcript: list[Message]) -> list[str]:
    """Paths of every file a successful edit_file/create_file call
    actually wrote, in first-occurrence order, deduplicated."""
    paths: list[str] = []
    seen: set[str] = set()
    for message in transcript:
        if message.role != "tool":
            continue
        content = message.content
        path = None
        if content.startswith(_EDIT_FILE_PREFIX) and content.endswith("."):
            path = content[len(_EDIT_FILE_PREFIX):-1]
        else:
            for prefix in _CREATE_FILE_PREFIXES:
                if content.startswith(prefix) and content.endswith("."):
                    path = content[len(prefix):-1]
                    break
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths
