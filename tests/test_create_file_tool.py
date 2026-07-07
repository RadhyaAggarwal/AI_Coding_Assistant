from tools.create_file import CreateFileError, CreateFileTool
from tools.path_safety import PathOutsideProjectError

import pytest


def test_creates_new_file(tmp_path):
    tool = CreateFileTool(tmp_path)
    result = tool.run(path="new.py", content="print('hi')\n")
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert "Created" in result


def test_creates_file_in_new_subdirectory(tmp_path):
    tool = CreateFileTool(tmp_path)
    tool.run(path="pkg/mod.py", content="x = 1\n")
    assert (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"


def test_replaces_existing_file_entirely(tmp_path):
    """The behavior edit_file used to (unsafely) provide via empty
    search: this tool's whole purpose is replacing a file's entire
    content, unlike edit_file which can only ever touch an exact
    matched substring."""
    (tmp_path / "existing.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    tool = CreateFileTool(tmp_path)
    result = tool.run(path="existing.py", content="def bar():\n    return 2\n")
    assert (tmp_path / "existing.py").read_text(encoding="utf-8") == "def bar():\n    return 2\n"
    assert "Replaced" in result


def test_refuses_content_that_would_break_python_syntax(tmp_path):
    """Reproduces the exact live failure: a model double-escaped a
    docstring's quotes when embedding Python into create_file's JSON
    'content' argument (writing \\\" where \" was correct) -- valid JSON
    either way, so this can only be caught by checking the decoded
    result, which is what corrupted the real file live. Must be refused,
    leaving any pre-existing file untouched."""
    (tmp_path / "existing.py").write_text("original content\n", encoding="utf-8")
    tool = CreateFileTool(tmp_path)
    corrupted = 'def is_palindrome(text):\n    \\"\\"\\"doc\\"\\"\\"\n    return text\n'

    with pytest.raises(CreateFileError, match="not valid"):
        tool.run(path="existing.py", content=corrupted)

    assert (tmp_path / "existing.py").read_text(encoding="utf-8") == "original content\n"


def test_refuses_content_that_would_break_css_syntax(tmp_path):
    tool = CreateFileTool(tmp_path)
    with pytest.raises(CreateFileError, match="not valid"):
        tool.run(path="style.css", content=".button { color: ")


def test_blocks_path_traversal(tmp_path):
    tool = CreateFileTool(tmp_path)
    with pytest.raises(PathOutsideProjectError):
        tool.run(path="../escape.py", content="evil")


def test_requires_confirmation_flag_set(tmp_path):
    assert CreateFileTool(tmp_path).requires_confirmation is True


def test_target_path_resolves_within_root(tmp_path):
    tool = CreateFileTool(tmp_path)
    resolved = tool.target_path({"path": "sample.py", "content": ""})
    assert resolved == (tmp_path / "sample.py").resolve()


def test_confirmation_message_generic_for_new_file(tmp_path):
    tool = CreateFileTool(tmp_path)
    message = tool.confirmation_message({"path": "new.py", "content": "x = 1\n"})
    assert "new.py" in message
    assert "REPLACE" not in message


def test_confirmation_message_warns_when_overwriting_existing_file(tmp_path):
    """This is the whole point of overriding confirmation_message: the
    raw arguments alone (a path and some content) don't make it obvious
    that approving this call discards everything currently in the file —
    the warning must say so explicitly and give real line counts, not
    just echo the call like the default message would."""
    (tmp_path / "existing.py").write_text("a\nb\nc\n", encoding="utf-8")
    tool = CreateFileTool(tmp_path)

    message = tool.confirmation_message({"path": "existing.py", "content": "x\ny\n"})

    assert "existing.py" in message
    assert "REPLACE" in message
    assert "3" in message  # current line count
    assert "2" in message  # new line count
