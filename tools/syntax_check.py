"""Pre-write syntax validation shared by edit_file and create_file.

Observed live: a model double-escaped a docstring's quotes when embedding
Python source into a JSON string argument (writing \\\" where \" was
correct) -- valid JSON either way, so nothing at the parsing layer could
have caught it, but the decoded content was no longer valid Python. That
content got written straight to disk, breaking the file with a real
SyntaxError. This mirrors the same defensive principle already used for
tool-call JSON elsewhere in this project: validate the result and reject
it with a clear, specific error before acting on it, rather than writing
something broken and finding out later.

Covers every language this project already parses elsewhere (CLAUDE.md:
ast for Python, tree-sitter for everything else), not just Python -- a
JS/CSS/HTML file gets the same protection, using the identical grammars
repo_index/ already depends on. Files of any other extension (or one
with no dedicated checker, e.g. no TypeScript grammar is installed) are
left unvalidated, the same scope limit repo_index/ already has.
"""
from pathlib import Path
from typing import Callable

import tree_sitter_css
import tree_sitter_html
import tree_sitter_javascript
from tree_sitter import Language, Parser


class InvalidSyntaxError(Exception):
    """Raised when content about to be written would leave the file
    syntactically invalid in a language this project can check -- the
    real parse error is included so the model has a concrete, specific
    signal to correct rather than a generic rejection.
    """


def _check_python(content: str) -> None:
    # compile(), not ast.parse() -- live-observed the difference matters:
    # ast.parse() only builds a syntax tree and accepts anything
    # *grammatically* well-formed, but Python enforces several rules only
    # at compile time, not parse time (a bare `return`/`yield` outside a
    # function, `break`/`continue` outside a loop, `nonlocal` with no
    # matching enclosing binding). A model's edit left `return x * 2` at
    # module level (an indentation slip merging a line into the wrong
    # scope) and ast.parse() accepted it silently -- the file was written
    # and genuinely couldn't be imported, exactly the corruption this
    # check exists to prevent, through a gap that had been there since
    # this file was first written, just never exercised by an error of
    # this specific shape before. compile(..., mode="exec") performs the
    # same full validation the real interpreter does before ever running
    # the file, and still raises the same SyntaxError type this function
    # already catches -- no other code here needs to change.
    compile(content, "<content>", "exec")


def _make_tree_sitter_checker(language_module) -> Callable[[str], None]:
    language = Language(language_module.language())

    def _check(content: str) -> None:
        parser = Parser(language)
        tree = parser.parse(content.encode("utf-8"))
        if tree.root_node.has_error:
            raise SyntaxError("tree-sitter found a parse error somewhere in the file")

    return _check


_CHECKERS: dict[str, Callable[[str], None]] = {
    ".py": _check_python,
    ".js": _make_tree_sitter_checker(tree_sitter_javascript),
    ".jsx": _make_tree_sitter_checker(tree_sitter_javascript),
    ".css": _make_tree_sitter_checker(tree_sitter_css),
    ".html": _make_tree_sitter_checker(tree_sitter_html),
}


def check_syntax(path: Path, content: str) -> None:
    """Raise InvalidSyntaxError if content is invalid for path's language.

    Silently does nothing for extensions with no checker registered --
    unrecognized/unsupported languages are simply not validated, not
    rejected.
    """
    checker = _CHECKERS.get(path.suffix)
    if checker is None:
        return
    try:
        checker(content)
    except SyntaxError as exc:
        raise InvalidSyntaxError(
            f"The content is not valid {path.suffix} syntax and was not "
            f"written: {exc}. Check for issues like mismatched quotes "
            '(e.g. an over-escaped docstring producing literal \\" '
            'instead of "), unbalanced brackets/tags, or bad indentation '
            "from a partial edit."
        )
