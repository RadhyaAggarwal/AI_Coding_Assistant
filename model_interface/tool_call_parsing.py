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
_ARGS_KEY_PATTERN = re.compile(r'"(?:' + "|".join(_ARGS_KEYS) + r')"\s*:')


def mentions_tool_call_attempt(text: str) -> bool:
    """True if a candidate JSON blob in text names *anything* under a
    name-like key (via the same convention extract_tool_calls()
    recognizes) AND supplies something under an arguments-like key —
    regardless of whether the blob is valid JSON, whether that value is
    the right shape, or whether the name matches a tool that actually
    exists. That combination is the actual signature of a genuine (if
    broken) tool-call attempt, as opposed to text that merely happens to
    mention something with no accompanying arguments at all, which is
    far more likely incidental than a real attempt.

    Does NOT require the name to match a real, registered tool. An
    earlier version did, and that turned out to be a real live gap, not
    just theoretical caution: given no tools left in budget, a model
    responded with prose plus a clearly tool-call-shaped blob naming
    "write_file" -- not a real tool this project has (it has
    create_file, not write_file) -- with genuine (if invalid, due to
    unescaped nested triple-quotes) arguments. Because the name didn't
    match a known tool, this function said "not an attempt," and the
    whole broken blob was handed to the user as if it were a clean
    answer -- exactly the failure this function exists to prevent, just
    via a hallucinated name instead of a malformed shape. Dropping the
    known-tool-name requirement doesn't reopen either of the two bugs
    this function was originally built to fix (see below) -- both are
    independently excluded by the name-key-exists and arguments-key-
    exists requirements alone, with no dependency on the name being real.

    Requiring *both* signals (a name-like key AND an arguments-like key)
    exists because either one alone is wrong in a different direction,
    both found live on the same night:
      - Name alone is too loose: asked to summarize a YAML file, the
        model answered by re-emitting its contents as a plain,
        syntactically valid JSON object with fields like "endpoint_url"
        and "model_name" — no name/tool/tool_name key at all, so this
        alone was already excluded by an earlier version requiring a
        name match, but a *different* case slipped through the other
        way: a genuine final answer that happened to mention a real
        tool's name in JSON-ish shape with no arguments key at all (e.g.
        `{"name": "read_file", "note": "..."}`) resolved as a fully
        valid, executable call with an empty-defaulted argument dict,
        which caused a real answer to be discarded. Requiring an
        arguments-like key too means an incidental mention (which has no
        reason to include one) doesn't count, while an actual attempt
        (which supplies one, even wrong-shaped) does.
      - Parseability alone is too loose in the other direction: a model
        can write syntactically *valid* JSON that names a real tool but
        gets the shape wrong, e.g. `{"name": "read_file", "arguments":
        "config.yaml"}` (a bare string instead of an object) — this
        parses fine but extract_tool_calls() rejects it (arguments isn't
        a dict), and a version of this function that only reacted to
        JSON *parse failures* missed it entirely, silently returning the
        broken JSON to the user as if it were a real answer.

    Two distinct call sites in agent_controller.loop.run() rely on this:
    the per-step branch only ever reaches it after extract_tool_calls()
    already found nothing resolvable, so here a positive result always
    means "attempted but broken/malformed," not "fully valid." The final
    step-budget-exhausted fallback needs the broader question — "is the
    model still trying to invoke a tool at all," even for an attempt that
    WOULD otherwise resolve cleanly — since no tools are offered at that
    point, so any such attempt is itself evidence something is still
    fixated on tool-call syntax rather than answering.
    """
    for blob in _candidate_json_blobs(text):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            if _NAME_KEY_PATTERN.search(blob) and _ARGS_KEY_PATTERN.search(blob):
                return True
            continue

        if not isinstance(parsed, dict):
            continue
        name = _first_present(parsed, _NAME_KEYS)
        if not isinstance(name, str):
            continue
        if _first_present(parsed, _ARGS_KEYS) is not None:
            return True

    return False
