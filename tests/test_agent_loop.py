"""End-to-end tests for agent_controller.loop.run against fake models,
covering the real failure mode observed against Ollama: a model that
writes its tool call as plain text instead of using the structured
tool_calls field.
"""
from model_interface.base import ModelInterface, ModelResponse
from tools.read_file import ReadFileTool
from tools.registry import ToolRegistry

from agent_controller.loop import run


class TextOnlyToolCallModel(ModelInterface):
    """Mimics the observed qwen2.5-coder:7b behavior: the tool call is
    written as plain JSON text in the response content, never populating
    the native tool_calls field."""

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt, tools=None):
        self.prompts.append(prompt)
        if "Observation:" in prompt:
            return ModelResponse(text="FINAL ANSWER USING OBSERVATION")
        return ModelResponse(
            text='{"name": "read_file", "arguments": {"path": "sample.txt"}}'
        )


def test_recovers_text_only_tool_call_and_executes_it(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = TextOnlyToolCallModel()

    answer = run("Summarize sample.txt", model, tools)

    assert answer == "FINAL ANSWER USING OBSERVATION"
    assert len(model.prompts) == 2
    assert "sample contents" in model.prompts[1]  # observation was fed back


class FailThenSucceedModel(ModelInterface):
    """First attempt omits the required 'path' argument; second attempt
    (after seeing the failure) supplies it correctly."""

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt, tools=None):
        self.prompts.append(prompt)
        if "Observation:" in prompt:
            return ModelResponse(text="FINAL ANSWER")
        if "That failed:" in prompt:
            return ModelResponse(
                text='{"name": "read_file", "arguments": {"path": "sample.txt"}}'
            )
        return ModelResponse(text='{"name": "read_file", "arguments": {}}')


def test_retries_after_validation_failure(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = FailThenSucceedModel()

    answer = run("Summarize sample.txt", model, tools)

    assert answer == "FINAL ANSWER"
    assert len(model.prompts) == 3


class AlwaysInvalidModel(ModelInterface):
    def __init__(self):
        self.call_count = 0

    def generate(self, prompt, tools=None):
        self.call_count += 1
        return ModelResponse(text='{"name": "read_file", "arguments": {}}')


def test_gives_up_after_max_attempts(tmp_path):
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = AlwaysInvalidModel()

    answer = run("Summarize sample.txt", model, tools)

    assert "Could not complete the request" in answer
    assert model.call_count == 2
