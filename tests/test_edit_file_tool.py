import pytest

from tools.edit_file import EditFileError, EditFileTool
from tools.path_safety import PathOutsideProjectError


def test_creates_new_file(tmp_path):
    tool = EditFileTool(tmp_path)
    result = tool.run(path="new.py", search="", replace="print('hi')\n")
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert "Created" in result


def test_refuses_create_when_file_already_exists(tmp_path):
    (tmp_path / "existing.py").write_text("x = 1\n", encoding="utf-8")
    tool = EditFileTool(tmp_path)
    with pytest.raises(EditFileError):
        tool.run(path="existing.py", search="", replace="y = 2\n")


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
    with pytest.raises(EditFileError):
        tool.run(path="missing.py", search="something", replace="else")


def test_blocks_path_traversal(tmp_path):
    tool = EditFileTool(tmp_path)
    with pytest.raises(PathOutsideProjectError):
        tool.run(path="../escape.py", search="", replace="evil")


def test_requires_confirmation_flag_set(tmp_path):
    assert EditFileTool(tmp_path).requires_confirmation is True


def test_target_path_resolves_within_root(tmp_path):
    tool = EditFileTool(tmp_path)
    resolved = tool.target_path({"path": "sample.py", "search": "", "replace": ""})
    assert resolved == (tmp_path / "sample.py").resolve()
