from repo_index.js_index import index_javascript_file


def test_indexes_function_and_class_declarations(tmp_path):
    source = (
        "function greet() {\n"
        "  return 'hi';\n"
        "}\n"
        "\n"
        "class Widget {\n"
        "  render() {\n"
        "    return null;\n"
        "  }\n"
        "}\n"
    )
    path = tmp_path / "sample.js"
    path.write_text(source, encoding="utf-8")

    result = index_javascript_file(path, "sample.js")

    assert result.language == "javascript"
    names_kinds = {(s.name, s.kind) for s in result.symbols}
    assert ("greet", "function") in names_kinds
    assert ("Widget", "class") in names_kinds
    assert ("render", "method") in names_kinds


def test_no_symbols_in_empty_file(tmp_path):
    path = tmp_path / "empty.js"
    path.write_text("", encoding="utf-8")

    result = index_javascript_file(path, "empty.js")

    assert result.symbols == []
