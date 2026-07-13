import pytest

from tools.path_safety import PathOutsideProjectError
from tools.search_code import SearchCodeTool


def test_finds_matching_line(tmp_path):
    (tmp_path / "sample.py").write_text("def login():\n    pass\n", encoding="utf-8")
    tool = SearchCodeTool(tmp_path)

    result = tool.run(query="def login")

    assert "sample.py:1:" in result
    assert "def login" in result


def test_case_insensitive(tmp_path):
    (tmp_path / "sample.py").write_text("HELLO WORLD\n", encoding="utf-8")
    tool = SearchCodeTool(tmp_path)
    result = tool.run(query="hello world")
    assert "sample.py:1:" in result


def test_no_matches(tmp_path):
    (tmp_path / "sample.py").write_text("nothing here\n", encoding="utf-8")
    tool = SearchCodeTool(tmp_path)
    assert tool.run(query="needle") == "No matches found."


def test_skips_junk_directories(tmp_path):
    skip_dir = tmp_path / "node_modules"
    skip_dir.mkdir()
    (skip_dir / "lib.js").write_text("findme", encoding="utf-8")
    tool = SearchCodeTool(tmp_path)
    assert tool.run(query="findme") == "No matches found."


def test_skips_agent_state_directory(tmp_path):
    """Regression test: search_code used to keep its own separate skip-dir
    list instead of reusing repo_index.scanner's, and that copy was
    missing ".agent_state" -- live-observed real consequence once
    --continue and the session log started writing conversation history
    there: a genuine match got buried under noise from the agent's own
    past conversation turns, which the model never acted on."""
    state_dir = tmp_path / ".agent_state"
    state_dir.mkdir()
    (state_dir / "last_conversation.json").write_text('{"content": "findme"}', encoding="utf-8")
    tool = SearchCodeTool(tmp_path)
    assert tool.run(query="findme") == "No matches found."


def test_blocks_path_traversal(tmp_path):
    tool = SearchCodeTool(tmp_path)
    with pytest.raises(PathOutsideProjectError):
        tool.run(query="x", path="..")
