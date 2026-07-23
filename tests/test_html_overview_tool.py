import pytest

from tools.html_overview import HtmlOverviewTool


def test_reports_scripts_and_stylesheets(tmp_path):
    (tmp_path / "index.html").write_text(
        '<html><head><link rel="stylesheet" href="styles.css">'
        '<script src="app.js"></script></head><body></body></html>\n',
        encoding="utf-8",
    )
    tool = HtmlOverviewTool(tmp_path)

    result = tool.run(path="index.html")

    assert "styles.css" in result
    assert "app.js" in result


def test_reports_none_for_plain_file(tmp_path):
    (tmp_path / "index.html").write_text("<div>hi</div>\n", encoding="utf-8")
    tool = HtmlOverviewTool(tmp_path)

    result = tool.run(path="index.html")

    assert "Stylesheets: none" in result
    assert "Scripts (external): none" in result


def test_missing_file_raises(tmp_path):
    tool = HtmlOverviewTool(tmp_path)
    with pytest.raises(FileNotFoundError):
        tool.run(path="missing.html")


def test_missing_file_error_suggests_a_real_close_match(tmp_path):
    """See tools/path_suggestions.py -- found missing by a post-session
    coherence audit: every other not-found-path tool (read_file,
    edit_file, list_directory, search_code) already got this, but
    html_overview was overlooked when it was built."""
    real = tmp_path / "core" / "templates" / "core" / "note_list.html"
    real.parent.mkdir(parents=True)
    real.write_text("<p>hi</p>", encoding="utf-8")
    tool = HtmlOverviewTool(tmp_path)

    with pytest.raises(FileNotFoundError, match="core/templates/core/note_list.html"):
        tool.run(path="templates/notes.html")


def test_missing_file_error_has_no_suggestion_text_when_nothing_matches(tmp_path):
    tool = HtmlOverviewTool(tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        tool.run(path="missing.html")
    assert "Did you mean" not in str(exc_info.value)
