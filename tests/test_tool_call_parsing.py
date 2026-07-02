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
