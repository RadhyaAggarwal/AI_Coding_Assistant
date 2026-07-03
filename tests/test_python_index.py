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


def test_indexes_plain_and_from_imports(tmp_path):
    source = (
        "import os.path\n"
        "from repo_index.models import Symbol\n"
        "from . import sibling\n"
    )
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")

    result = index_python_file(path, "sample.py")

    imported = {i.imported for i in result.imports}
    assert "os.path" in imported
    assert "repo_index.models" in imported
    assert "." in imported


def test_indexes_call_sites_including_method_calls(tmp_path):
    source = (
        "def outer():\n"
        "    helper()\n"
        "    obj.method()\n"
        "    self.selfcall()\n"
    )
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")

    result = index_python_file(path, "sample.py")

    callees = {c.callee_name for c in result.calls}
    assert "helper" in callees
    assert "method" in callees
    assert "selfcall" in callees


def test_call_site_line_numbers(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def outer():\n    helper()\n", encoding="utf-8")

    result = index_python_file(path, "sample.py")

    [call] = result.calls
    assert call.callee_name == "helper"
    assert call.line == 2
