"""Lightweight, append-only log of past requests and their outcomes, so
a later request can answer "what did we work on before" without
replaying a full conversation.

Deliberately NOT a transcript (see conversation_store.py, scoped to
short-term --continue) -- full-fidelity replay would grow unboundedly
across weeks of use and go stale as files change underneath it. This is
a thin, bounded pointer instead: one entry per request (timestamp, a
truncated version of what was asked, a truncated outcome summary, and
whether it completed), meant to jog memory or ground tools/recent_activity.py's
lookup -- never to be trusted as a substitute for checking real current
project state (git log, the files themselves). Entries are capped at
_MAX_RETAINED_SESSIONS, oldest dropped first, so this stays a minimal,
well-understood log rather than growing forever.
"""
import json
import time
from pathlib import Path

_FILENAME = "session_log.json"
_MAX_RETAINED_SESSIONS = 500
_MAX_FIELD_CHARS = 200


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_FIELD_CHARS:
        return text
    return text[:_MAX_FIELD_CHARS] + "..."


def append_session(state_dir: Path, request: str, answer: str, completed: bool) -> None:
    entries = read_sessions(state_dir)
    entries.append(
        {
            "timestamp": time.time(),
            "request": _truncate(request),
            "outcome": _truncate(answer),
            "completed": completed,
        }
    )
    entries = entries[-_MAX_RETAINED_SESSIONS:]

    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / _FILENAME
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def read_sessions(state_dir: Path, limit: int | None = None) -> list[dict]:
    """Returns [] if nothing was ever logged, or the file is unreadable/
    corrupt -- treated as "nothing there" rather than raising, the same
    convention conversation_store.py uses.
    """
    path = state_dir / _FILENAME
    if not path.is_file():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if limit is not None:
        entries = entries[-limit:]
    return entries
