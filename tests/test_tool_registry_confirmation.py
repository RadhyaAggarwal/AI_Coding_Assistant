import pytest

from tools.base import Tool
from tools.confirmation import ToolCallDeniedError
from tools.registry import ToolRegistry


class _FakeRiskyTool(Tool):
    name = "fake_risky"
    description = "test tool"
    parameters = {"type": "object", "properties": {}, "required": []}
    requires_confirmation = True

    def __init__(self):
        self.ran = False

    def run(self, **kwargs):
        self.ran = True
        return "done"


class _FakeSafeTool(Tool):
    name = "fake_safe"
    description = "test tool"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        return "ok"


def test_executes_when_confirmed():
    tool = _FakeRiskyTool()
    registry = ToolRegistry(confirm=lambda description: True)
    registry.register(tool)

    result = registry.execute("fake_risky", {})

    assert result == "done"
    assert tool.ran is True


def test_denies_when_not_confirmed():
    tool = _FakeRiskyTool()
    registry = ToolRegistry(confirm=lambda description: False)
    registry.register(tool)

    with pytest.raises(ToolCallDeniedError):
        registry.execute("fake_risky", {})

    assert tool.ran is False


def test_read_only_tool_never_prompts():
    def _explode(_description):
        raise AssertionError("confirm() should not be called for a safe tool")

    registry = ToolRegistry(confirm=_explode)
    registry.register(_FakeSafeTool())

    assert registry.execute("fake_safe", {}) == "ok"


def test_confirmation_required_names_lists_only_confirmation_gated_tools():
    registry = ToolRegistry(confirm=lambda description: True)
    registry.register(_FakeRiskyTool())
    registry.register(_FakeSafeTool())

    assert registry.confirmation_required_names() == {"fake_risky"}
