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


def test_missing_file_error_suggests_a_real_close_match(tmp_path):
    """See tools/path_suggestions.py -- reproduces the exact live gap of
    a wrong path guess never finding the one real file that existed."""
    real = tmp_path / "core" / "templates" / "core" / "note_list.html"
    real.parent.mkdir(parents=True)
    real.write_text("<p>hi</p>", encoding="utf-8")
    tool = ReadFileTool(tmp_path)

    with pytest.raises(FileNotFoundError, match="core/templates/core/note_list.html"):
        tool.run(path="templates/notes.html")


def test_missing_file_error_has_no_suggestion_text_when_nothing_matches(tmp_path):
    tool = ReadFileTool(tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        tool.run(path="missing.txt")
    assert "Did you mean" not in str(exc_info.value)


def test_progress_message_names_the_file(tmp_path):
    tool = ReadFileTool(tmp_path)
    assert tool.progress_message({"path": "hello.txt"}) == "Reading hello.txt..."
