from repo_index.css_index import index_css_file


def test_indexes_class_and_id_selectors(tmp_path):
    source = (
        ".button {\n"
        "  color: red;\n"
        "}\n"
        "\n"
        "#header {\n"
        "  color: blue;\n"
        "}\n"
    )
    path = tmp_path / "sample.css"
    path.write_text(source, encoding="utf-8")

    result = index_css_file(path, "sample.css")

    assert result.language == "css"
    names_kinds = {(s.name, s.kind) for s in result.symbols}
    assert (".button", "selector") in names_kinds
    assert ("#header", "selector") in names_kinds


def test_indexes_keyframes(tmp_path):
    source = "@keyframes fade {\n  from { opacity: 0; }\n  to { opacity: 1; }\n}\n"
    path = tmp_path / "sample.css"
    path.write_text(source, encoding="utf-8")

    result = index_css_file(path, "sample.css")

    names_kinds = {(s.name, s.kind) for s in result.symbols}
    assert ("fade", "keyframes") in names_kinds


def test_indexes_comma_separated_selectors(tmp_path):
    path = tmp_path / "sample.css"
    path.write_text(".card, .panel {\n  border: 1px solid black;\n}\n", encoding="utf-8")

    result = index_css_file(path, "sample.css")

    names = {s.name for s in result.symbols}
    assert ".card" in names
    assert ".panel" in names


def test_no_symbols_in_empty_file(tmp_path):
    path = tmp_path / "empty.css"
    path.write_text("", encoding="utf-8")

    result = index_css_file(path, "empty.css")

    assert result.symbols == []
