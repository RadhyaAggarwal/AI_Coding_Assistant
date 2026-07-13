import pytest

from tools.edit_file import EditFileError, EditFileTool
from tools.path_safety import PathOutsideProjectError


def test_refuses_empty_search_on_existing_file(tmp_path):
    """The error must redirect to create_file, not just refuse — this is
    the exact live failure observed repeatedly: the model reaches for an
    empty 'search' against a file that already exists, and needs an
    actionable next step, not just a rejection. edit_file no longer has
    any create/replace-whole-file mode at all, unlike its old behavior."""
    (tmp_path / "existing.py").write_text("x = 1\n", encoding="utf-8")
    tool = EditFileTool(tmp_path)
    with pytest.raises(EditFileError, match="create_file"):
        tool.run(path="existing.py", search="", replace="y = 2\n")
    assert (tmp_path / "existing.py").read_text(encoding="utf-8") == "x = 1\n"


def test_refuses_empty_search_on_missing_file(tmp_path):
    """edit_file never creates a file either, regardless of 'search' --
    a missing target is refused before 'search' is even considered."""
    tool = EditFileTool(tmp_path)
    with pytest.raises(EditFileError, match="create_file"):
        tool.run(path="new.py", search="", replace="print('hi')\n")
    assert not (tmp_path / "new.py").exists()


def test_edits_unique_match(tmp_path):
    (tmp_path / "sample.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    tool = EditFileTool(tmp_path)
    result = tool.run(path="sample.py", search="return 1", replace="return 2")
    assert (tmp_path / "sample.py").read_text(encoding="utf-8") == "def foo():\n    return 2\n"
    assert "Edited" in result


def test_refuses_when_search_not_found(tmp_path):
    (tmp_path / "sample.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    tool = EditFileTool(tmp_path)
    with pytest.raises(EditFileError):
        tool.run(path="sample.py", search="not in file", replace="whatever")


def test_refuses_ambiguous_match(tmp_path):
    (tmp_path / "sample.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    tool = EditFileTool(tmp_path)
    with pytest.raises(EditFileError):
        tool.run(path="sample.py", search="x = 1", replace="x = 2")


def test_refuses_edit_when_target_missing(tmp_path):
    tool = EditFileTool(tmp_path)
    with pytest.raises(EditFileError, match="create_file"):
        tool.run(path="missing.py", search="something", replace="else")


def test_refuses_edit_that_would_break_python_syntax(tmp_path):
    """Reproduces the earlier live failure (a middle-ground complex-task
    test): replacing 'if value < max_val:' with a bare 'return ...'
    orphaned the following indented line with no block header to justify
    it -- a real IndentationError. Must be refused, leaving the file
    untouched, rather than writing broken content."""
    original = (
        "def f(value, max_val):\n"
        "    if value < max_val:\n"
        "        value = max_val\n"
        "    return value\n"
    )
    (tmp_path / "sample.py").write_text(original, encoding="utf-8")
    tool = EditFileTool(tmp_path)

    with pytest.raises(EditFileError, match="not valid"):
        tool.run(
            path="sample.py",
            search="if value < max_val:",
            replace="return value * 2",
        )

    assert (tmp_path / "sample.py").read_text(encoding="utf-8") == original


def test_refuses_edit_that_would_break_javascript_syntax(tmp_path):
    (tmp_path / "sample.js").write_text("function f() { return 1; }", encoding="utf-8")
    tool = EditFileTool(tmp_path)

    with pytest.raises(EditFileError, match="not valid"):
        tool.run(path="sample.js", search="return 1;", replace="if (x")


def test_blocks_path_traversal(tmp_path):
    tool = EditFileTool(tmp_path)
    with pytest.raises(PathOutsideProjectError):
        tool.run(path="../escape.py", search="x", replace="evil")


def test_requires_confirmation_flag_set(tmp_path):
    assert EditFileTool(tmp_path).requires_confirmation is True


def test_target_path_resolves_within_root(tmp_path):
    tool = EditFileTool(tmp_path)
    resolved = tool.target_path({"path": "sample.py", "search": "x", "replace": "y"})
    assert resolved == (tmp_path / "sample.py").resolve()


def test_confirmation_message_shows_a_real_diff_for_a_clean_match(tmp_path):
    (tmp_path / "sample.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    tool = EditFileTool(tmp_path)

    message = tool.confirmation_message({"path": "sample.py", "search": "return 1", "replace": "return 2"})

    assert "-    return 1" in message
    assert "+    return 2" in message


def test_confirmation_message_falls_back_when_search_not_found(tmp_path):
    (tmp_path / "sample.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    tool = EditFileTool(tmp_path)

    message = tool.confirmation_message({"path": "sample.py", "search": "not in file", "replace": "x"})

    assert "not in file" in message  # falls back to the generic raw-arguments message
    assert "@@" not in message  # no diff hunk markers -- no diff was shown


def test_confirmation_message_falls_back_when_file_does_not_exist(tmp_path):
    tool = EditFileTool(tmp_path)

    message = tool.confirmation_message({"path": "missing.py", "search": "x", "replace": "y"})

    assert "@@" not in message
