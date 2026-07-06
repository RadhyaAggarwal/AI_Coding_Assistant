"""Best-effort extraction of a tool call from free-form model text.

Small/local models are inconsistent about using a backend's structured
tool-calling mechanism — they sometimes just print the JSON they think a
tool call should look like as ordinary text instead of triggering it. This
is the fallback layer described in the Build Plan Addendum (2.2): defensive
parsing around every tool call, built in from the start rather than
assumed away.

This module only does syntactic recovery: is there a plausible
{name, arguments} object in the text, and does the name match one of the
tools the model was offered? It is a heuristic, not a full parser — it
won't recover from every malformed shape a model might produce. Argument
-level schema validation happens in tools/, closer to each tool's own
definition.
"""
import json
import re
from typing import Any

from model_interface.base import ToolCall

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_NAME_KEYS = ("name", "tool", "tool_name")
_ARGS_KEYS = ("arguments", "parameters", "input", "args")


def _find_balanced_json_objects(text: str) -> list[str]:
    """Every top-level {...} span in text, found via brace-depth counting
    rather than a greedy regex. A greedy r"\\{.*\\}" spans from the first
    "{" to the very last "}" in the whole text — if the model writes out
    more than one tool-call attempt back-to-back in a single response
    (observed live: it planned two calls ahead and printed both JSON
    objects one after another), that greedy span merges them into one
    invalid blob that fails to parse at all, silently discarding both
    attempts instead of recovering the first one.
    """
    blobs = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    blobs.append(text[start : i + 1])
                    start = None
    return blobs


def _candidate_json_blobs(text: str) -> list[str]:
    blobs = [match.group(1) for match in _CODE_FENCE_RE.finditer(text)]
    blobs.extend(_find_balanced_json_objects(text))
    return blobs


def _first_present(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return None


def extract_tool_calls(text: str, known_tool_names: set[str]) -> list[ToolCall]:
    """Look for every {name, arguments}-shaped JSON object naming a known
    tool, not just the first.

    Observed live: a model can legitimately plan several tool calls in one
    turn (e.g. a JSON array covering both parts of a compound request).
    Stopping at the first meant the model never learned its later calls
    didn't run, and would go on to report a false negative for whatever
    they would have found instead of noticing the gap. Duplicate calls
    (same name and arguments) are collapsed to one — a single code-fenced
    call is otherwise matched twice, once by the fence regex and once by
    the brace scan on the same span, and re-running an identical call
    within one turn carries no new information anyway.

    Returns an empty list if nothing plausible is found — callers should
    treat that as "the model didn't attempt a tool call," not as an error.
    """
    calls: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()
    for blob in _candidate_json_blobs(text):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue

        name = _first_present(parsed, _NAME_KEYS)
        if not isinstance(name, str) or name not in known_tool_names:
            continue

        arguments = _first_present(parsed, _ARGS_KEYS)
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            continue

        key = (name, json.dumps(arguments, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        calls.append(ToolCall(name=name, arguments=arguments))

    return calls


def extract_tool_call(text: str, known_tool_names: set[str]) -> ToolCall | None:
    """Look for a {name, arguments}-shaped JSON object naming a known tool.

    Returns None if nothing plausible is found — callers should treat that
    as "the model didn't attempt a tool call," not as an error. Kept
    alongside extract_tool_calls() (plural) for callers that only ever
    want the first/only call.
    """
    calls = extract_tool_calls(text, known_tool_names)
    return calls[0] if calls else None


def looks_like_unparsed_tool_call(text: str, known_tool_names: set[str]) -> bool:
    """True if text contains a brace-balanced {...} span that didn't
    resolve into any valid call — evidence the model was attempting a
    tool call and produced something malformed (e.g. an unescaped quote
    from an embedded docstring breaking the JSON string it sits in), not
    that it gave a genuine prose answer.

    Observed live: a model embedded a real docstring (with its own
    unescaped quote characters) inside a JSON "replace" argument, making
    the JSON invalid. extract_tool_calls() correctly discarded it (as it
    should — this module doesn't guess at malformed JSON), but the caller
    then had nothing to distinguish "no tool call was attempted" from
    "one was attempted and broke," and the raw broken JSON got returned
    to the user as if it were a final answer. This reuses the same
    brace-depth scan _candidate_json_blobs() already does — if it found a
    span at all, something was attempted, regardless of whether it went
    on to parse.
    """
    if extract_tool_calls(text, known_tool_names):
        return False
    return bool(_candidate_json_blobs(text))
