from tools.find_symbol import FindSymbolTool


def test_finds_python_function(tmp_path):
    (tmp_path / "main.py").write_text("def greet():\n    pass\n", encoding="utf-8")
    tool = FindSymbolTool(tmp_path)

    result = tool.run(name="greet")

    assert "main.py:1" in result
    assert "function greet" in result


def test_includes_real_source_line(tmp_path):
    (tmp_path / "main.py").write_text("def greet():\n    pass\n", encoding="utf-8")
    tool = FindSymbolTool(tmp_path)

    result = tool.run(name="greet")

    assert "def greet():" in result


def test_snippet_includes_body_not_just_signature(tmp_path):
    (tmp_path / "main.py").write_text(
        "def greet():\n    print('hi')\n    return 'hi'\n", encoding="utf-8"
    )
    tool = FindSymbolTool(tmp_path)

    result = tool.run(name="greet")

    assert "def greet():" in result
    assert "print('hi')" in result
    assert "return 'hi'" in result


def test_css_snippet_includes_declarations(tmp_path):
    (tmp_path / "styles.css").write_text(
        ".button {\n  color: red;\n  padding: 8px;\n}\n", encoding="utf-8"
    )
    tool = FindSymbolTool(tmp_path)

    result = tool.run(name=".button")

    assert "color: red;" in result
    assert "padding: 8px;" in result


def test_reports_no_match(tmp_path):
    tool = FindSymbolTool(tmp_path)
    result = tool.run(name="nonexistent")
    assert "No symbol named" in result
