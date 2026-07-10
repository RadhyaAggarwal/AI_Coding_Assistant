from tools.base import Tool
from tools.registry import ToolRegistry


class _FakeSafeTool(Tool):
    name = "fake_safe"
    description = "test tool"
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs):
        return "ok"


class _FakeToolWithCustomProgressMessage(Tool):
    name = "fake_custom"
    description = "test tool"
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def run(self, **kwargs):
        return "ok"

    def progress_message(self, arguments):
        return f"Doing something to {arguments['path']}..."


def test_default_progress_message_names_the_tool():
    tool = _FakeSafeTool()
    assert tool.progress_message({}) == "Running fake_safe..."


def test_registry_reports_progress_before_running_a_read_only_tool():
    seen = []
    registry = ToolRegistry(report=seen.append)
    registry.register(_FakeSafeTool())

    registry.execute("fake_safe", {})

    assert seen == ["Running fake_safe..."]


def test_registry_uses_a_tools_custom_progress_message():
    seen = []
    registry = ToolRegistry(report=seen.append)
    registry.register(_FakeToolWithCustomProgressMessage())

    registry.execute("fake_custom", {"path": "config.yaml"})

    assert seen == ["Doing something to config.yaml..."]


def test_registry_reports_progress_before_the_confirmation_prompt():
    class _FakeRiskyTool(Tool):
        name = "fake_risky"
        description = "test tool"
        parameters = {"type": "object", "properties": {}, "required": []}
        requires_confirmation = True

        def run(self, **kwargs):
            return "done"

    events = []
    registry = ToolRegistry(
        confirm=lambda description: (events.append(("confirm", description)), True)[1],
        report=lambda message: events.append(("report", message)),
    )
    registry.register(_FakeRiskyTool())

    registry.execute("fake_risky", {})

    assert events[0] == ("report", "Running fake_risky...")
    assert events[1][0] == "confirm"
