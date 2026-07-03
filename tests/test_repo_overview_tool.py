from tools.repo_overview import RepoOverviewTool


def test_reports_languages_and_manifests(tmp_path):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")

    tool = RepoOverviewTool(tmp_path)
    result = tool.run()

    assert "python" in result
    assert "requirements.txt" in result


def test_reports_none_found_for_empty_project(tmp_path):
    tool = RepoOverviewTool(tmp_path)
    result = tool.run()

    assert "none detected" in result
    assert "none found" in result
