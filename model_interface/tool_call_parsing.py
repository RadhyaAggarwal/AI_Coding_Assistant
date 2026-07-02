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
_BARE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_NAME_KEYS = ("name", "tool", "tool_name")
_ARGS_KEYS = ("arguments", "parameters", "input", "args")


def _candidate_json_blobs(text: str) -> list[str]:
    blobs = [match.group(1) for match in _CODE_FENCE_RE.finditer(text)]
    bare_match = _BARE_JSON_RE.search(text)
    if bare_match:
        blobs.append(bare_match.group(0))
    return blobs


def _first_present(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return None


def extract_tool_call(text: str, known_tool_names: set[str]) -> ToolCall | None:
    """Look for a {name, arguments}-shaped JSON object naming a known tool.

    Returns None if nothing plausible is found — callers should treat that
    as "the model didn't attempt a tool call," not as an error.
    """
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

        return ToolCall(name=name, arguments=arguments)

    return None
