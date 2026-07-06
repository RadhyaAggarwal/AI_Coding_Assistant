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


_NAME_KEY_PATTERN = re.compile(
    r'"(?:' + "|".join(_NAME_KEYS) + r')"\s*:\s*"([^"]*)"'
)


def looks_like_unparsed_tool_call(text: str, known_tool_names: set[str]) -> bool:
    """True only for a {...} span that fails to parse as JSON at all but
    still carries the literal signature of a tool-call attempt — a
    "name"/"tool"/"tool_name" key naming one of the offered tools,
    present in the raw text even though the surrounding JSON is broken.

    Observed live (the case this exists for): a model embedded a real
    docstring (with its own unescaped quote characters) inside a JSON
    "replace" argument, making the JSON invalid. extract_tool_calls()
    correctly discarded it, but the caller then had nothing to
    distinguish "no tool call was attempted" from "one was attempted and
    broke," and the raw broken JSON got returned to the user as if it
    were a final answer.

    An earlier version of this function treated *any* brace-balanced
    span with zero resolved calls as evidence of a broken attempt. That
    over-fired on a real, different live case: asked to summarize a YAML
    file, the model answered by re-emitting its contents as a plain
    (syntactically valid, unrelated-to-any-tool) JSON object — a
    legitimate, if unhelpfully-formatted, answer with no tool-call intent
    at all. Treating it as "broken JSON, please fix" sent the loop into
    a nudge/retry cycle the model had no way to resolve, since there was
    nothing to fix. Requiring the blob to both fail to parse *and* name
    a real tool distinguishes an actual failed attempt from ordinary
    prose that merely happens to contain balanced braces.
    """
    if extract_tool_calls(text, known_tool_names):
        return False
    for blob in _candidate_json_blobs(text):
        try:
            json.loads(blob)
            continue  # parsed fine -- just irrelevant JSON, not a broken attempt
        except json.JSONDecodeError:
            pass
        match = _NAME_KEY_PATTERN.search(blob)
        if match and match.group(1) in known_tool_names:
            return True
    return False
