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


def test_indexes_es_import_and_require(tmp_path):
    source = (
        'import { foo } from "./bar.js";\n'
        'import baz from "baz-lib";\n'
        'const x = require("legacy-module");\n'
    )
    path = tmp_path / "sample.js"
    path.write_text(source, encoding="utf-8")

    result = index_javascript_file(path, "sample.js")

    imported = {i.imported for i in result.imports}
    assert "./bar.js" in imported
    assert "baz-lib" in imported
    assert "legacy-module" in imported
    # require() calls are recorded as imports, not spurious "calls to require"
    assert "require" not in {c.callee_name for c in result.calls}


def test_indexes_call_sites_including_method_calls(tmp_path):
    source = (
        "function outer() {\n"
        "  helper();\n"
        "  obj.method();\n"
        "  this.selfCall();\n"
        "}\n"
    )
    path = tmp_path / "sample.js"
    path.write_text(source, encoding="utf-8")

    result = index_javascript_file(path, "sample.js")

    callees = {c.callee_name for c in result.calls}
    assert "helper" in callees
    assert "method" in callees
    assert "selfCall" in callees
