from tools.find_importers import FindImportersTool


def test_finds_python_importer(tmp_path):
    (tmp_path / "main.py").write_text("import os.path\n", encoding="utf-8")
    tool = FindImportersTool(tmp_path)

    result = tool.run(module_name="os.path")

    assert "main.py:1" in result
    assert "import os.path" in result


def test_finds_js_require_importer(tmp_path):
    (tmp_path / "main.js").write_text('const x = require("legacy-module");\n', encoding="utf-8")
    tool = FindImportersTool(tmp_path)

    result = tool.run(module_name="legacy-module")

    assert "main.js:1" in result


def test_reports_no_match(tmp_path):
    tool = FindImportersTool(tmp_path)
    result = tool.run(module_name="nonexistent")
    assert "No file imports" in result
