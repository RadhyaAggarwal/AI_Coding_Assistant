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
