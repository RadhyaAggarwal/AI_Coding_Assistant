from repo_index.html_index import html_references, index_html_file

_SAMPLE = """<!DOCTYPE html>
<html>
<head>
  <title>Test</title>
  <link rel="stylesheet" href="styles.css">
  <script src="app.js"></script>
</head>
<body>
  <div id="header" class="main">
    <h1 id="title">Hello</h1>
  </div>
  <script>
    console.log("inline");
  </script>
</body>
</html>
"""


def test_indexes_elements_with_id_as_symbols(tmp_path):
    path = tmp_path / "index.html"
    path.write_text(_SAMPLE, encoding="utf-8")

    result = index_html_file(path, "index.html")

    assert result.language == "html"
    names_kinds = {(s.name, s.kind) for s in result.symbols}
    assert ("#header", "element") in names_kinds
    assert ("#title", "element") in names_kinds


def test_elements_without_id_are_not_indexed(tmp_path):
    path = tmp_path / "index.html"
    path.write_text("<div class=\"no-id\">hi</div>\n", encoding="utf-8")

    result = index_html_file(path, "index.html")

    assert result.symbols == []


def test_html_references_finds_external_script_and_stylesheet(tmp_path):
    path = tmp_path / "index.html"
    path.write_text(_SAMPLE, encoding="utf-8")

    refs = html_references(path)

    assert refs.scripts == ["app.js"]
    assert refs.stylesheets == ["styles.css"]
    assert refs.inline_script_count == 1
    assert refs.element_count > 0


def test_html_references_no_external_refs(tmp_path):
    path = tmp_path / "index.html"
    path.write_text("<div>hi</div>\n", encoding="utf-8")

    refs = html_references(path)

    assert refs.scripts == []
    assert refs.stylesheets == []
    assert refs.inline_script_count == 0
