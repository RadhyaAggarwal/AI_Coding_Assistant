"""End-to-end tests for agent_controller.loop.run against fake models.

Covers the real failure mode observed against Ollama (tool call written as
plain text instead of using the structured tool_calls field), the bounded
multi-step chaining added afterward, and the defenses that make chaining
safe (repeated-call refusal, a hard step budget). Fake models branch on
message-list shape (length/roles) rather than string content, since the
conversation is a proper role-separated history, not a flattened string.
"""
import json

from model_interface.base import Message, ModelInterface, ModelResponse
from tools.edit_file import EditFileTool
from tools.read_file import ReadFileTool
from tools.registry import ToolRegistry
from tools.search_code import SearchCodeTool

from agent_controller.loop import _MAX_STEPS, run


def _call_text(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments})


class HugeObservationModel(ModelInterface):
    """Calls a tool that returns a very large observation, then
    answers — verifies the loop caps it before it enters the
    conversation (context_budget.cap_observation), regardless of
    whether the tool itself self-limits (read_file doesn't)."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        if len(messages) == 2:
            return ModelResponse(text=_call_text("read_file", {"path": "huge.txt"}))
        return ModelResponse(text="Got it.")


def test_huge_tool_observation_gets_capped(tmp_path):
    (tmp_path / "huge.txt").write_text("x" * 10000, encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = HugeObservationModel()

    answer = run("Summarize huge.txt", model, tools)

    assert answer == "Got it."
    tool_message = model.calls[-1][-1]
    assert tool_message.role == "tool"
    assert len(tool_message.content) < 10000
    assert "truncated" in tool_message.content


class RepeatedLargeReadModel(ModelInterface):
    """Reads several different large files in sequence, then answers —
    used to verify that once accumulated history exceeds a tight budget,
    the message list actually sent to the model stops growing without
    bound, rather than the loop just accumulating forever."""

    def __init__(self):
        self.sent_message_counts: list[int] = []
        self._step = 0

    def generate(self, messages, tools=None):
        self.sent_message_counts.append(len(messages))
        if self._step < 5:
            path = f"file{self._step}.txt"
            self._step += 1
            return ModelResponse(text=_call_text("read_file", {"path": path}))
        return ModelResponse(text="Done.")


def test_trim_to_budget_keeps_sent_history_bounded_under_tight_budget(tmp_path):
    for i in range(5):
        (tmp_path / f"file{i}.txt").write_text("x" * 4000, encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = RepeatedLargeReadModel()

    answer = run("Read all the files", model, tools, context_window_tokens=2100)

    assert answer == "Done."
    # Without trimming, the raw history would keep growing every call
    # (2, 4, 6, 8, 10, 12). With a tight budget, later calls should not
    # reach that untrimmed size.
    untrimmed_final_size = 2 + 2 * 5
    assert model.sent_message_counts[-1] < untrimmed_final_size


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


class EditVerifyFixVerifyModel(ModelInterface):
    """Simulates a realistic fix cycle: create a buggy file, verify by
    reading it, fix it, verify again, then answer. 4 tool-call turns plus
    a final answer turn — exercises the headroom _MAX_STEPS=6 was raised
    to support (4 was only enough for a single tool call)."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        step = len([m for m in messages if m.role == "tool"])
        if step == 0:
            return ModelResponse(
                text=_call_text(
                    "edit_file",
                    {"path": "calc.py", "search": "", "replace": "def add(a, b):\n    return a - b\n"},
                )
            )
        if step == 1:
            return ModelResponse(text=_call_text("read_file", {"path": "calc.py"}))
        if step == 2:
            return ModelResponse(
                text=_call_text(
                    "edit_file",
                    {"path": "calc.py", "search": "return a - b", "replace": "return a + b"},
                )
            )
        if step == 3:
            return ModelResponse(text=_call_text("read_file", {"path": "calc.py"}))
        return ModelResponse(text="Fixed: add() now returns a + b.")


def test_completes_realistic_edit_verify_fix_cycle_within_budget(tmp_path):
    tools = ToolRegistry(confirm=lambda description: True)  # edit_file requires confirmation
    tools.register(EditFileTool(tmp_path))
    tools.register(ReadFileTool(tmp_path))
    model = EditVerifyFixVerifyModel()

    answer = run("Fix the add function in calc.py", model, tools)

    assert answer == "Fixed: add() now returns a + b."
    assert len(model.calls) == 5  # 4 tool-call turns + 1 final answer turn
    assert "return a + b" in (tmp_path / "calc.py").read_text(encoding="utf-8")
