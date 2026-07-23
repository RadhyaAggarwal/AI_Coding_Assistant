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


class _FakeToolThatShowsItsResult(Tool):
    name = "fake_shows_result"
    description = "test tool"
    parameters = {"type": "object", "properties": {}, "required": []}
    show_result = True

    def run(self, **kwargs):
        return "the real result"


def test_default_show_result_is_false():
    assert _FakeSafeTool().show_result is False


def test_registry_reports_the_result_when_show_result_is_true():
    seen = []
    registry = ToolRegistry(report=seen.append)
    registry.register(_FakeToolThatShowsItsResult())

    result = registry.execute("fake_shows_result", {})

    assert result == "the real result"
    assert seen == ["Running fake_shows_result...", "the real result"]


def test_registry_does_not_report_the_result_when_show_result_is_false():
    seen = []
    registry = ToolRegistry(report=seen.append)
    registry.register(_FakeSafeTool())

    registry.execute("fake_safe", {})

    assert seen == ["Running fake_safe..."]  # the "ok" result is never reported


def test_default_should_show_result_matches_the_static_show_result_flag():
    assert _FakeSafeTool().should_show_result("anything") is False
    assert _FakeToolThatShowsItsResult().should_show_result("anything") is True


class _FakeToolThatConditionallyShowsItsResult(Tool):
    """Mirrors edit_file/create_file: show_result stays False (the
    ordinary case is silent, unchanged from before), but
    should_show_result overrides per-call based on the real content."""

    name = "fake_conditional"
    description = "test tool"
    parameters = {"type": "object", "properties": {"noteworthy": {"type": "boolean"}}, "required": []}

    def run(self, noteworthy=False):
        return "Note: something" if noteworthy else "ok"

    def should_show_result(self, result):
        return result.startswith("Note:")


def test_registry_uses_should_show_result_not_the_static_flag():
    seen = []
    registry = ToolRegistry(report=seen.append)
    registry.register(_FakeToolThatConditionallyShowsItsResult())

    registry.execute("fake_conditional", {"noteworthy": False})
    assert seen == ["Running fake_conditional..."]  # ordinary case: silent, static flag is False

    seen.clear()
    registry.execute("fake_conditional", {"noteworthy": True})
    assert seen == ["Running fake_conditional...", "Note: something"]  # this call's real result is shown
