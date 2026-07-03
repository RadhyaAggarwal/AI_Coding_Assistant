from repo_index.python_index import index_python_file


def test_indexes_functions_and_classes(tmp_path):
    source = (
        "def top_level():\n"
        "    pass\n"
        "\n"
        "class Foo:\n"
        "    def method(self):\n"
        "        pass\n"
    )
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")

    result = index_python_file(path, "sample.py")

    assert result.language == "python"
    names_kinds = {(s.name, s.kind) for s in result.symbols}
    assert ("top_level", "function") in names_kinds
    assert ("Foo", "class") in names_kinds
    assert ("method", "method") in names_kinds


def test_indexes_async_functions(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("async def fetch():\n    pass\n", encoding="utf-8")

    result = index_python_file(path, "sample.py")

    [symbol] = result.symbols
    assert symbol.name == "fetch"
    assert symbol.kind == "function"


def test_records_correct_line_numbers(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("\n\ndef foo():\n    pass\n", encoding="utf-8")

    result = index_python_file(path, "sample.py")

    [symbol] = result.symbols
    assert symbol.line == 3


def test_raises_on_invalid_syntax(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("def foo(:\n", encoding="utf-8")

    import pytest
    with pytest.raises(SyntaxError):
        index_python_file(path, "broken.py")
