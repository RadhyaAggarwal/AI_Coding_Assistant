from tools.diff_preview import colorize_diff, content_preview, unified_diff_preview


def test_shows_added_and_removed_lines():
    old = "line one\nline two\nline three\n"
    new = "line one\nline TWO CHANGED\nline three\n"

    diff = unified_diff_preview(old, new, "sample.py")

    assert "-line two" in diff
    assert "+line TWO CHANGED" in diff
    assert "sample.py (before)" in diff
    assert "sample.py (after)" in diff


def test_no_difference_returns_a_plain_note():
    content = "same content\n"
    assert unified_diff_preview(content, content, "sample.py") == "(no textual difference)"


def test_caps_a_very_long_diff():
    old = "\n".join(f"old line {i}" for i in range(200)) + "\n"
    new = "\n".join(f"new line {i}" for i in range(200)) + "\n"

    diff = unified_diff_preview(old, new, "sample.py")

    assert "more diff line(s) omitted" in diff
    assert len(diff.splitlines()) < 200


def test_content_preview_returns_short_content_unchanged():
    content = "line one\nline two\n"
    assert content_preview(content) == content


def test_content_preview_caps_long_content():
    content = "\n".join(f"line {i}" for i in range(50))

    preview = content_preview(content, max_lines=20)

    assert "more line(s) omitted" in preview
    assert len(preview.splitlines()) <= 21  # 20 real lines + 1 summary line


def test_colorize_diff_adds_color_when_stdout_is_a_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    diff = unified_diff_preview("line one\nline two\n", "line one\nline TWO\n", "sample.py")

    colored = colorize_diff(diff)

    assert "\x1b[31m-line two" in colored
    assert "\x1b[32m+line TWO" in colored
    assert "\x1b[36m@@" in colored
    # the original diff content is still present, unaltered, inside the
    # color codes -- a substring check against the plain diff must still
    # work for anything that only cares about the text, not the color
    assert "-line two" in colored
    assert "+line TWO" in colored


def test_colorize_diff_is_a_noop_when_stdout_is_not_a_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    diff = unified_diff_preview("line one\nline two\n", "line one\nline TWO\n", "sample.py")

    assert colorize_diff(diff) == diff
    assert "\x1b[" not in colorize_diff(diff)


def test_colorize_diff_leaves_file_headers_uncolored(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    diff = unified_diff_preview("a\n", "b\n", "sample.py")

    colored = colorize_diff(diff)

    assert "\x1b[32msample.py (after)" not in colored
    assert "sample.py (before)" in colored
    assert "sample.py (after)" in colored
