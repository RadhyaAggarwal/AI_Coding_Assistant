import pytest

from tools.read_file import PathOutsideProjectError, ReadFileTool


def test_reads_file_within_project(tmp_path):
    (tmp_path / "hello.txt").write_text("hi there", encoding="utf-8")
    tool = ReadFileTool(tmp_path)
    assert tool.run(path="hello.txt") == "hi there"


def test_blocks_path_traversal(tmp_path):
    tool = ReadFileTool(tmp_path)
    with pytest.raises(PathOutsideProjectError):
        tool.run(path="../outside.txt")


def test_missing_file_raises(tmp_path):
    tool = ReadFileTool(tmp_path)
    with pytest.raises(FileNotFoundError):
        tool.run(path="missing.txt")
