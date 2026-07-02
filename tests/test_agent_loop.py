"""End-to-end tests for agent_controller.loop.run against fake models.

Covers the real failure mode observed against Ollama (tool call written as
plain text instead of using the structured tool_calls field), the bounded
multi-step chaining added afterward, and the defenses that make chaining
safe (repeated-call refusal, a hard step budget). Fake models branch on
message-list shape (length/roles) rather than string content, since the
conversation is a proper role-separated history, not a flattened string.
"""
from model_interface.base import Message, ModelInterface, ModelResponse
from tools.read_file import ReadFileTool
from tools.registry import ToolRegistry
from tools.search_code import SearchCodeTool

from agent_controller.loop import _MAX_STEPS, run


class TextOnlyToolCallModel(ModelInterface):
    """Mimics the observed qwen2.5-coder:7b behavior: the tool call is
    written as plain JSON text in the response content, never populating
    the native tool_calls field."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))  # snapshot: loop mutates this list in place
        if len(messages) == 2:  # system + user: first turn, no history yet
            return ModelResponse(
                text='{"name": "read_file", "arguments": {"path": "sample.txt"}}'
            )
        return ModelResponse(text="FINAL ANSWER USING OBSERVATION")


def test_recovers_text_only_tool_call_and_executes_it(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = TextOnlyToolCallModel()

    answer = run("Summarize sample.txt", model, tools)

    assert answer == "FINAL ANSWER USING OBSERVATION"
    assert len(model.calls) == 2
    final_call_messages = model.calls[1]
    assert [m.role for m in final_call_messages] == ["system", "user", "assistant", "tool"]
    assert final_call_messages[-1].content == "sample contents"  # observation fed back verbatim


class FailThenSucceedModel(ModelInterface):
    """First attempt omits the required 'path' argument; second attempt
    (after seeing the failure) supplies it correctly."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))  # snapshot: loop mutates this list in place
        if len(messages) == 2:
            return ModelResponse(text='{"name": "read_file", "arguments": {}}')
        if len(messages) == 4:
            return ModelResponse(
                text='{"name": "read_file", "arguments": {"path": "sample.txt"}}'
            )
        return ModelResponse(text="FINAL ANSWER")


def test_retries_after_validation_failure(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = FailThenSucceedModel()

    answer = run("Summarize sample.txt", model, tools)

    assert answer == "FINAL ANSWER"
    assert len(model.calls) == 3
    # the retry should have seen the validation error as a tool-role message
    retry_messages = model.calls[1]
    assert retry_messages[-1].role == "tool"
    assert "Missing required argument" in retry_messages[-1].content


class AlwaysInvalidModel(ModelInterface):
    def __init__(self):
        self.call_count = 0

    def generate(self, messages, tools=None):
        self.call_count += 1
        return ModelResponse(text='{"name": "read_file", "arguments": {}}')


def test_gives_up_after_max_steps(tmp_path):
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = AlwaysInvalidModel()

    answer = run("Summarize sample.txt", model, tools)

    assert f"within {_MAX_STEPS} steps" in answer
    # _MAX_STEPS failed attempts, plus one final no-tools synthesis call
    # that also fails to produce a clean answer (guarded, not leaked raw)
    assert model.call_count == _MAX_STEPS + 1


class MultiStepModel(ModelInterface):
    """Calls two different tools in sequence before answering — exercises
    the new multi-step chaining capability."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(text='{"name": "search_code", "arguments": {"query": "login"}}')
        if len(tool_messages) == 1:
            return ModelResponse(text='{"name": "read_file", "arguments": {"path": "auth.py"}}')
        return ModelResponse(text="FINAL ANSWER FROM TWO TOOLS")


def test_chains_multiple_different_tool_calls(tmp_path):
    (tmp_path / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(SearchCodeTool(tmp_path))
    tools.register(ReadFileTool(tmp_path))
    model = MultiStepModel()

    answer = run("Find the login function and explain it", model, tools)

    assert answer == "FINAL ANSWER FROM TWO TOOLS"
    assert len(model.calls) == 3
    final_call_messages = model.calls[-1]
    assert [m.role for m in final_call_messages] == [
        "system", "user", "assistant", "tool", "assistant", "tool",
    ]


class RepeatsCallModel(ModelInterface):
    """Calls read_file, then tries the exact same call again instead of
    answering — the loop should refuse the repeat, not re-run the tool or
    loop forever."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) <= 1:
            return ModelResponse(text='{"name": "read_file", "arguments": {"path": "sample.txt"}}')
        return ModelResponse(text="FINAL ANSWER AFTER REPEAT WARNING")


def test_refuses_identical_repeated_tool_call(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = RepeatsCallModel()

    answer = run("Summarize sample.txt", model, tools)

    assert answer == "FINAL ANSWER AFTER REPEAT WARNING"
    assert len(model.calls) == 3
    # the repeat should have been refused with a warning, not re-executed
    # (which would have put "sample contents" there again instead)
    third_call_messages = model.calls[2]
    assert third_call_messages[-1].role == "tool"
    assert "already called this exact tool" in third_call_messages[-1].content
