from model_interface.tool_call_parsing import extract_tool_call


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
