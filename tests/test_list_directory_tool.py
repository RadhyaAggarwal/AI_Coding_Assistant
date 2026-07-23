import pytest

from tools.list_directory import ListDirectoryTool
from tools.path_safety import PathOutsideProjectError


def test_lists_files_and_dirs(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    tool = ListDirectoryTool(tmp_path)

    result = tool.run(path=".")

    assert "f  a.txt" in result
    assert "d  subdir" in result


def test_empty_directory(tmp_path):
    tool = ListDirectoryTool(tmp_path)
    assert tool.run(path=".") == "(empty directory)"


def test_blocks_path_traversal(tmp_path):
    tool = ListDirectoryTool(tmp_path)
    with pytest.raises(PathOutsideProjectError):
        tool.run(path="..")


def test_missing_directory_raises(tmp_path):
    tool = ListDirectoryTool(tmp_path)
    with pytest.raises(NotADirectoryError):
        tool.run(path="does_not_exist")


def test_missing_directory_error_suggests_a_real_close_match(tmp_path):
    """See tools/path_suggestions.py."""
    (tmp_path / "core" / "templates").mkdir(parents=True)
    tool = ListDirectoryTool(tmp_path)

    with pytest.raises(NotADirectoryError, match="core/templates"):
        tool.run(path="template")


def test_missing_directory_error_has_no_suggestion_text_when_nothing_matches(tmp_path):
    tool = ListDirectoryTool(tmp_path)
    with pytest.raises(NotADirectoryError) as exc_info:
        tool.run(path="does_not_exist")
    assert "Did you mean" not in str(exc_info.value)
