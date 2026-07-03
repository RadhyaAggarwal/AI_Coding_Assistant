from repo_index.scanner import scan


def test_scan_detects_languages_and_manifests(tmp_path):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "app.js").write_text("let x = 1;\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")

    summary = scan(tmp_path)

    assert summary.file_count == 3
    assert summary.languages == {"python": 1, "javascript": 1}
    assert summary.manifests == ["requirements.txt"]


def test_scan_skips_junk_directories(tmp_path):
    skip_dir = tmp_path / "node_modules"
    skip_dir.mkdir()
    (skip_dir / "lib.js").write_text("x", encoding="utf-8")
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    summary = scan(tmp_path)

    assert summary.file_count == 1
    assert summary.languages == {"python": 1}


def test_scan_empty_project(tmp_path):
    summary = scan(tmp_path)
    assert summary.file_count == 0
    assert summary.languages == {}
    assert summary.manifests == []
