import json

from model_interface.tool_call_parsing import (
    extract_tool_call,
    extract_tool_calls,
    mentions_tool_call_attempt,
)


def test_extracts_bare_json_tool_call():
    text = '{"name": "read_file", "arguments": {"path": "config.yaml"}}'
    call = extract_tool_call(text, {"read_file"})
    assert call is not None
    assert call.name == "read_file"
    assert call.arguments == {"path": "config.yaml"}


def test_extracts_tool_call_from_code_fence():
    text = (
        "Sure, let me check that.\n"
        '```json\n{"name": "read_file", "arguments": {"path": "config.yaml"}}\n```\n'
    )
    call = extract_tool_call(text, {"read_file"})
    assert call is not None
    assert call.name == "read_file"


def test_accepts_key_aliases():
    text = '{"tool": "read_file", "parameters": {"path": "config.yaml"}}'
    call = extract_tool_call(text, {"read_file"})
    assert call is not None
    assert call.arguments == {"path": "config.yaml"}


def test_returns_none_for_unknown_tool_name():
    text = '{"name": "delete_everything", "arguments": {}}'
    assert extract_tool_call(text, {"read_file"}) is None


def test_returns_none_for_plain_prose():
    assert extract_tool_call("Sure, here's a summary of the file.", {"read_file"}) is None


def test_returns_none_for_malformed_json():
    assert extract_tool_call('{"name": "read_file", "arguments": ', {"read_file"}) is None


def test_recovers_first_call_when_model_writes_two_back_to_back():
    """Reproduces a real failure: the model planned two tool calls ahead
    and printed both JSON objects one after another in a single response.
    A greedy \\{.*\\} regex would span both into one invalid blob and
    recover nothing; this must recover the first one."""
    text = (
        '{"name": "find_importers", "arguments": {"module_name": "repo_index.indexer"}}\n\n'
        '{"name": "find_callers", "arguments": {"name": "cap_observation"}}'
    )
    call = extract_tool_call(text, {"find_importers", "find_callers"})
    assert call is not None
    assert call.name == "find_importers"
    assert call.arguments == {"module_name": "repo_index.indexer"}


def test_recovers_first_call_when_model_writes_three_back_to_back():
    """Not a 2-object special case: the brace-depth scan finds every
    top-level {...} span in one pass over the whole text, regardless of
    how many there are."""
    text = (
        '{"name": "find_importers", "arguments": {"module_name": "a"}}\n\n'
        '{"name": "find_callers", "arguments": {"name": "b"}}\n\n'
        '{"name": "read_file", "arguments": {"path": "c"}}'
    )
    call = extract_tool_call(text, {"find_importers", "find_callers", "read_file"})
    assert call is not None
    assert call.name == "find_importers"


def test_recovers_call_with_nested_object_argument():
    text = '{"name": "edit_file", "arguments": {"path": "a.py", "meta": {"nested": true}}}'
    call = extract_tool_call(text, {"edit_file"})
    assert call is not None
    assert call.arguments == {"path": "a.py", "meta": {"nested": True}}


def test_extract_tool_calls_returns_every_call_in_a_json_array():
    """Reproduces the exact live shape: the model plans both parts of a
    compound request at once as a JSON array of two calls."""
    text = json.dumps(
        [
            {"name": "find_importers", "arguments": {"module_name": "agent_controller.tool_router"}},
            {"name": "find_callers", "arguments": {"name": "snippet_at"}},
        ]
    )
    calls = extract_tool_calls(text, {"find_importers", "find_callers"})
    assert [c.name for c in calls] == ["find_importers", "find_callers"]
    assert calls[0].arguments == {"module_name": "agent_controller.tool_router"}
    assert calls[1].arguments == {"name": "snippet_at"}


def test_extract_tool_calls_returns_every_call_when_written_back_to_back():
    text = (
        '{"name": "find_importers", "arguments": {"module_name": "a"}}\n\n'
        '{"name": "find_callers", "arguments": {"name": "b"}}\n\n'
        '{"name": "read_file", "arguments": {"path": "c"}}'
    )
    calls = extract_tool_calls(text, {"find_importers", "find_callers", "read_file"})
    assert [c.name for c in calls] == ["find_importers", "find_callers", "read_file"]


def test_extract_tool_calls_dedupes_identical_call_matched_twice():
    """A single code-fenced call is matched both by the fence regex and by
    the brace scan over the same span — that must collapse to one call,
    not silently double-execute it."""
    text = (
        "Sure, let me check that.\n"
        '```json\n{"name": "read_file", "arguments": {"path": "config.yaml"}}\n```\n'
    )
    calls = extract_tool_calls(text, {"read_file"})
    assert len(calls) == 1


def test_extract_tool_calls_returns_empty_list_for_plain_prose():
    assert extract_tool_calls("Sure, here's a summary of the file.", {"read_file"}) == []


def test_extract_tool_call_still_returns_only_the_first_of_several():
    text = (
        '{"name": "find_importers", "arguments": {"module_name": "a"}}\n\n'
        '{"name": "find_callers", "arguments": {"name": "b"}}'
    )
    call = extract_tool_call(text, {"find_importers", "find_callers"})
    assert call is not None
    assert call.name == "find_importers"


def test_mentions_tool_call_attempt_true_for_malformed_json():
    """Reproduces the exact live shape: a docstring's own unescaped
    quotes, embedded inside a JSON string value, break the JSON -- this
    must be recognized as an attempted-but-failed call, not plain prose."""
    text = (
        '{"name": "edit_file", "arguments": {"path": "a.py", "search": "", '
        '"replace": "def f():\\n    """docstring"""\\n    return 1"}}'
    )
    assert mentions_tool_call_attempt(text) is True


def test_mentions_tool_call_attempt_false_for_plain_prose():
    assert mentions_tool_call_attempt("Sure, here's a summary.") is False


def test_mentions_tool_call_attempt_true_when_a_valid_call_is_present():
    """Unlike the per-step call site (which only ever reaches this
    function after extract_tool_calls() already found nothing, so a True
    here always means "broken"), the final step-budget-exhausted fallback
    needs this to also be True for a fully valid, resolvable call --
    since no tools are offered at that point, any such attempt (broken or
    not) means the model is still fixated on tool-call syntax rather than
    answering, and that fallback shouldn't hand the raw JSON to the user
    either way."""
    text = '{"name": "read_file", "arguments": {"path": "a.py"}}'
    assert mentions_tool_call_attempt(text) is True


def test_mentions_tool_call_attempt_false_for_valid_unrelated_json():
    """Reproduces a real live regression: asked to summarize a YAML file,
    the model answered by re-emitting its contents as a plain,
    syntactically valid JSON object with no tool-call intent at all (no
    name/tool/tool_name key) -- this must not be mistaken for a broken
    tool-call attempt, or the loop nudges the model to "fix" JSON that
    was never broken and was never a tool call to begin with."""
    text = (
        "```json\n"
        "{\n"
        '  "endpoint_url": "http://localhost:11434",\n'
        '  "model_name": "qwen2.5-coder:7b"\n'
        "}\n"
        "```"
    )
    assert mentions_tool_call_attempt(text) is False


def test_mentions_tool_call_attempt_true_when_broken_json_names_an_unrecognized_tool():
    """Reproduces a real live regression, distinct from (and found after)
    the other cases here: given no tools left in budget, a model
    responded with prose plus a clearly tool-call-shaped blob naming
    "write_file" -- not a real tool this project has (it has create_file,
    not write_file) -- with genuine (if invalid, due to unescaped nested
    triple-quotes) arguments. An earlier version of this function
    required the name to match a *known* tool, so it concluded "not a
    real attempt" and let the whole broken blob through to the user as if
    it were a clean answer. A hallucinated tool name with a real
    arguments payload is still unambiguously an attempt -- it's broken by
    naming something that doesn't exist, the same way other cases here
    are broken by malformed JSON or a wrong-shaped argument, and should
    be treated the same way, not treated as a normal answer."""
    text = (
        '{"name": "delete_everything", "arguments": {"path": "a.py", '
        '"extra": "def f():\\n    """doc"""\\n"}}'
    )
    assert mentions_tool_call_attempt(text) is True


def test_mentions_tool_call_attempt_true_for_valid_json_with_wrong_shaped_arguments():
    """Reproduces a real gap found in a post-session audit: valid JSON
    naming a real tool, but "arguments" is a bare string instead of an
    object -- extract_tool_calls() correctly rejects this (wrong shape),
    but a version of this function that only reacted to JSON parse
    *failures* missed it entirely, since this parses fine. The model
    still clearly named a real tool and supplied something under an
    arguments-like key, which is enough to call it an attempt even though
    it never fails to parse."""
    text = '{"name": "read_file", "arguments": "config.yaml"}'
    assert mentions_tool_call_attempt(text) is True


def test_mentions_tool_call_attempt_false_for_tool_name_mentioned_with_no_arguments_key():
    """Reproduces a real gap found in a post-session audit: a genuine
    final answer that happens to mention a real tool's name in JSON-ish
    shape, with no arguments key at all, previously resolved as a fully
    valid call with an empty-defaulted argument dict (extract_tool_calls
    defaults a missing arguments key to {}) -- which discarded a real
    answer at the final-fallback call site. Requiring an arguments-like
    key too means an incidental mention, which has no reason to include
    one, doesn't count."""
    text = '{"name": "read_file", "note": "the config mentions read_file settings"}'
    assert mentions_tool_call_attempt(text) is False
