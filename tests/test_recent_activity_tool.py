from agent_controller.session_log import append_session
from tools.recent_activity import RecentActivityTool


def test_returns_a_placeholder_when_nothing_logged(tmp_path):
    tool = RecentActivityTool(tmp_path)
    assert tool.run() == "No recorded past sessions."


def test_returns_logged_sessions_with_request_and_outcome(tmp_path):
    append_session(tmp_path, "fix the pricing bug", "Fixed apply_discount.", completed=True)
    tool = RecentActivityTool(tmp_path)

    result = tool.run()

    assert "fix the pricing bug" in result
    assert "Fixed apply_discount." in result
    assert "[done]" in result


def test_marks_an_incomplete_session_distinctly(tmp_path):
    append_session(tmp_path, "do something hard", "Could not complete the request within 6 steps.", completed=False)
    tool = RecentActivityTool(tmp_path)

    result = tool.run()

    assert "[incomplete]" in result


def test_respects_the_limit_argument(tmp_path):
    for i in range(5):
        append_session(tmp_path, f"request {i}", "outcome", completed=True)
    tool = RecentActivityTool(tmp_path)

    result = tool.run(limit=2)

    assert "request 3" in result
    assert "request 4" in result
    assert "request 0" not in result


def test_progress_message_is_generic():
    tool = RecentActivityTool(".")
    assert tool.progress_message({}) == "Checking recent activity..."
