from tools.source_snippet import snippet_at


def test_returns_window_starting_at_line(tmp_path):
    (tmp_path / "sample.py").write_text("a\nb\nc\nd\n", encoding="utf-8")

    result = snippet_at(tmp_path, "sample.py", 2)

    assert "b" in result
    assert "c" in result
    assert "a" not in result


def test_marks_truncation_when_more_lines_remain(tmp_path):
    lines = "\n".join(str(i) for i in range(20))
    (tmp_path / "sample.py").write_text(lines, encoding="utf-8")

    result = snippet_at(tmp_path, "sample.py", 1)

    assert "..." in result


def test_returns_unavailable_for_missing_file(tmp_path):
    result = snippet_at(tmp_path, "missing.py", 1)
    assert "unavailable" in result


def test_returns_unavailable_for_out_of_range_line(tmp_path):
    (tmp_path / "sample.py").write_text("only one line\n", encoding="utf-8")
    result = snippet_at(tmp_path, "sample.py", 50)
    assert "unavailable" in result
