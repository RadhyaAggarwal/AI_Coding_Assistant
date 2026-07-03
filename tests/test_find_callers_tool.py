from tools.find_callers import FindCallersTool


def test_finds_python_call_site(tmp_path):
    (tmp_path / "main.py").write_text("def use():\n    helper()\n", encoding="utf-8")
    tool = FindCallersTool(tmp_path)

    result = tool.run(name="helper")

    assert "main.py:2" in result
    assert "helper()" in result


def test_finds_method_call_regardless_of_object(tmp_path):
    (tmp_path / "main.py").write_text("def use():\n    obj.render()\n", encoding="utf-8")
    tool = FindCallersTool(tmp_path)

    result = tool.run(name="render")

    assert "main.py:2" in result


def test_reports_no_match(tmp_path):
    tool = FindCallersTool(tmp_path)
    result = tool.run(name="nonexistent")
    assert "No calls to" in result
