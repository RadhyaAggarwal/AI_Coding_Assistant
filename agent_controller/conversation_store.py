"""Persists a conversation's message history -- and the agent loop's
duplicate-call memory -- to disk so a follow-up request can pick up where
a previous one left off (see main.py's `--continue` flag).

Deliberately full-fidelity -- the exact prior Messages, not a summary --
since this is meant for continuing very soon after an interrupted or
partial task, where a lossy summary would throw away detail a follow-up
might actually need (the same reasoning agent_controller/loop.py's
`transcript` parameter is built on). This is NOT meant for a hypothetical
cross-day session log; that would need a different, bounded design to
avoid growing forever and going stale as files change -- see project
memory / DEPLOYMENT.md discussion for that separate, not-yet-built idea.

already_called is persisted alongside messages, not derived from them on
load -- live-observed gap: a model can re-run an identical, already-
answered tool call (e.g. the same search_code query) across several
separate --continue invocations, each one a fresh run() call whose
in-memory dedup guard starts empty, since agent_controller/loop.py's
already_called set previously had no memory of calls made in a prior
--continue turn. Re-deriving it by re-parsing saved message text back
into calls would duplicate loop.py's own tool-call parsing and its
clear-on-confirmation-gated-success rule (see loop.py's run() docstring)
in a second, easy-to-drift-out-of-sync place; persisting the exact set
run() already maintains avoids that.

Only ever holds the single most recent conversation -- there is no
multi-session management here, by design, to keep this a minimal, well-
understood MVP rather than a bigger subsystem nobody asked for yet.
"""
import json
from pathlib import Path

from model_interface.base import Message

_FILENAME = "last_conversation.json"


def save_conversation(
    state_dir: Path,
    messages: list[Message],
    already_called: set[tuple[str, str]] = frozenset(),
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / _FILENAME
    payload = {
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "already_called": [[name, args_json] for name, args_json in already_called],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_conversation(
    state_dir: Path,
) -> tuple[list[Message], set[tuple[str, str]]] | None:
    """Returns None if there's nothing to continue -- no prior
    conversation was ever saved, or the saved file is unreadable/corrupt/
    an older pre-already_called format (all treated the same as "nothing
    there" rather than raising, since a damaged state file shouldn't
    block starting a fresh request)."""
    path = state_dir / _FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = [Message(role=item["role"], content=item["content"]) for item in payload["messages"]]
        already_called = {tuple(pair) for pair in payload["already_called"]}
        return messages, already_called
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        return None
