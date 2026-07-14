from pathlib import Path

import pytest

from tools.syntax_check import InvalidSyntaxError, check_syntax


def test_accepts_valid_python():
    check_syntax(Path("x.py"), "def f():\n    return 1\n")


def test_rejects_invalid_python():
    with pytest.raises(InvalidSyntaxError):
        check_syntax(Path("x.py"), "def f(:\n    return 1\n")


def test_rejects_return_outside_a_function():
    """Reproduces the exact live failure: ast.parse() only builds a
    syntax tree and accepts anything grammatically well-formed --
    'return' outside a function is only caught by Python's compiler,
    not its parser. A model's edit merged a line into the wrong scope
    (an indentation slip), and this check silently accepted content
    that genuinely could not be imported, until fixed to use compile()
    instead of ast.parse()."""
    corrupted = "call_log = []\ndef f(x):\n    call_log.append(x)\nreturn x * 2\n"
    with pytest.raises(InvalidSyntaxError, match="not valid .py syntax"):
        check_syntax(Path("x.py"), corrupted)


def test_rejects_the_exact_live_double_escaped_docstring_corruption():
    """Reproduces the exact live failure: a model double-escaped a
    docstring's quotes when embedding Python into a JSON string argument
    (writing \\\" where \" was correct) -- valid JSON either way, so this
    can only be caught by checking the decoded result, not the JSON."""
    corrupted = 'def f():\n    \\"\\"\\"doc\\"\\"\\"\n    return 1\n'
    with pytest.raises(InvalidSyntaxError, match="not valid .py syntax"):
        check_syntax(Path("x.py"), corrupted)


def test_accepts_valid_javascript():
    check_syntax(Path("x.js"), "function f() { return 1; }")


def test_rejects_invalid_javascript():
    with pytest.raises(InvalidSyntaxError):
        check_syntax(Path("x.js"), "function f() { return 1; \n if (x")


def test_accepts_valid_css():
    check_syntax(Path("x.css"), ".button { color: red; }")


def test_rejects_invalid_css():
    with pytest.raises(InvalidSyntaxError):
        check_syntax(Path("x.css"), ".button { color: ")


def test_accepts_valid_html():
    check_syntax(Path("x.html"), '<div id="x">hi</div>')


def test_rejects_invalid_html():
    with pytest.raises(InvalidSyntaxError):
        check_syntax(Path("x.html"), '<div id="x"')


def test_leaves_unsupported_extensions_unvalidated():
    """No TypeScript grammar is installed (matching repo_index/, which
    doesn't index .ts either) -- an unsupported extension is silently
    left unchecked, not rejected."""
    check_syntax(Path("x.ts"), "this is not valid anything (((")


def test_jsx_uses_the_javascript_checker():
    with pytest.raises(InvalidSyntaxError):
        check_syntax(Path("x.jsx"), "function f() { return 1; \n if (x")
