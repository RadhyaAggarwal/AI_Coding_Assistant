from repo_index.indexer import RepoIndex


def test_indexes_python_and_javascript_files(tmp_path):
    (tmp_path / "main.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (tmp_path / "app.js").write_text("function bar() {}\n", encoding="utf-8")

    index = RepoIndex(tmp_path)

    assert set(index.files) == {"main.py", "app.js"}
    assert index.files["main.py"].language == "python"
    assert index.files["app.js"].language == "javascript"


def test_indexes_css_files(tmp_path):
    (tmp_path / "styles.css").write_text(".button {\n  color: red;\n}\n", encoding="utf-8")

    index = RepoIndex(tmp_path)

    assert "styles.css" in index.files
    assert index.files["styles.css"].language == "css"
    assert index.find_symbol(".button")[0].file == "styles.css"


def test_indexes_html_files(tmp_path):
    (tmp_path / "index.html").write_text(
        '<div id="header">hi</div>\n', encoding="utf-8"
    )

    index = RepoIndex(tmp_path)

    assert "index.html" in index.files
    assert index.files["index.html"].language == "html"
    assert index.find_symbol("#header")[0].file == "index.html"


def test_find_symbol_case_insensitive(tmp_path):
    (tmp_path / "main.py").write_text("def Foo():\n    pass\n", encoding="utf-8")

    index = RepoIndex(tmp_path)
    matches = index.find_symbol("foo")

    assert len(matches) == 1
    assert matches[0].name == "Foo"
    assert matches[0].file == "main.py"


def test_find_symbol_no_match(tmp_path):
    (tmp_path / "main.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    index = RepoIndex(tmp_path)
    assert index.find_symbol("does_not_exist") == []


def test_skips_files_with_syntax_errors(tmp_path):
    (tmp_path / "broken.py").write_text("def foo(:\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("def bar():\n    pass\n", encoding="utf-8")

    index = RepoIndex(tmp_path)

    assert "broken.py" not in index.files
    assert "ok.py" in index.files


def test_ignores_unsupported_extensions(tmp_path):
    (tmp_path / "notes.md").write_text("# hello\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("def foo():\n    pass\n", encoding="utf-8")

    index = RepoIndex(tmp_path)

    assert set(index.files) == {"main.py"}
