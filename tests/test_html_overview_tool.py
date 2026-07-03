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
