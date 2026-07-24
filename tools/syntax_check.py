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
import ast
import builtins
import re
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


# Live-observed real failure, not hypothetical: a model wrote
# <a href='{% url 'delete_note' note.id %}'> -- valid, idiomatic Django
# template syntax (the template engine resolves {% %} before any HTML
# parsing happens), but tree-sitter's plain HTML grammar has no concept
# of template tags, sees the tag's own matching quote as closing the
# outer attribute early, and correctly (from its own narrow view) calls
# the rest malformed. This never changes what gets accepted or rejected
# -- check_syntax()'s blocking behavior stays exactly as certain as it
# already was -- it only adds a specific, mechanically-detected hint to
# an already-correct rejection, the same "ground with a real fact"
# pattern as the rest of this project's error messages. Scoped to the
# tree-sitter-checked languages only; Python's compile() errors are a
# different, unrelated failure shape.
_TEMPLATE_TAG = re.compile(r"\{%.*?%\}|\{\{.*?\}\}", re.DOTALL)
_TEMPLATE_TAG_EXTENSIONS = frozenset({".html", ".js", ".jsx", ".css"})


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
        message = (
            f"The content is not valid {path.suffix} syntax and was not "
            f"written: {exc}. Check for issues like mismatched quotes "
            '(e.g. an over-escaped docstring producing literal \\" '
            'instead of "), unbalanced brackets/tags, or bad indentation '
            "from a partial edit."
        )
        if path.suffix in _TEMPLATE_TAG_EXTENSIONS and _TEMPLATE_TAG.search(content):
            message += (
                " This content also contains template-tag syntax (e.g. "
                "{% %} or {{ }}) -- if it uses the same quote character "
                "as the string/attribute it's nested inside (e.g. "
                "href='{% url 'x' %}'), that's a likely cause: this "
                "checker doesn't understand template syntax and sees the "
                "tag's own quote as closing the outer one early. Try a "
                "different quote style for one of them."
            )
        raise InvalidSyntaxError(message)


_BUILTIN_NAMES = frozenset(dir(builtins))
_TERMINATING_STATEMENTS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def _collect_bound_names(tree: ast.AST) -> set[str]:
    """Every name bound anywhere in the file, flattened across every
    scope -- deliberately not scope-aware. A real per-scope tracker
    (what a proper linter like pyflakes does) could tell a name that's
    genuinely undefined in *this* function apart from one that's merely
    defined in a different one -- but getting that wrong risks a false
    positive on code that's actually fine, which matters far more here
    than the coverage lost by being flat: a flat view can only under-
    flag (miss a real cross-scope bug), never wrongly flag something
    that's genuinely fine elsewhere in the file.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.MatchAs) and node.name:
            # match/case capture patterns (`case direction:`, `case [x,
            # direction]:`) bind a name as a plain string attribute, not
            # an ast.Name node -- invisible to the ast.Name/Store walk
            # above. Confirmed live: without this, a real, correctly-
            # bound capture like `direction` in `case ["go", direction]:`
            # was flagged as undefined.
            bound.add(node.name)
        elif isinstance(node, ast.MatchStar) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
    return bound


def _has_star_import(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
        for node in ast.walk(tree)
    )


def _find_undefined_names(tree: ast.AST) -> list[tuple[str, int]]:
    if _has_star_import(tree):
        # A star import binds an unknown set of names that only the real
        # module (not this file's AST) knows about -- e.g. a Django
        # settings/production.py doing `from .base import *` is
        # completely idiomatic and would make every name from base.py
        # look "undefined" to a checker that can't see into it. Skipping
        # the whole file rather than guessing keeps this from ever being
        # confidently wrong -- coverage lost for one file, not a false
        # positive risked on real, valid code.
        return []
    bound = _collect_bound_names(tree)
    seen: set[str] = set()
    undefined: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in bound and node.id not in _BUILTIN_NAMES and node.id not in seen:
                seen.add(node.id)
                undefined.append((node.id, node.lineno))
    return undefined


def _get_blocks(node: ast.AST) -> list[list[ast.stmt]]:
    """Every literal list-of-statements ("block") attached to this node
    that dead code could hide in."""
    blocks = []
    for field in ("body", "orelse", "finalbody"):
        value = getattr(node, field, None)
        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            blocks.append(value)
    if isinstance(node, ast.Try):
        for handler in node.handlers:
            blocks.append(handler.body)
    return blocks


def _find_dead_code(tree: ast.AST) -> list[int]:
    """Statements unreachable because they follow an unconditional
    return/raise/break/continue earlier in the same block -- deliberately
    narrow: only literal same-block sequential unreachability, never
    "does every branch of this if/else return" (a much harder, genuine
    control-flow-analysis problem this doesn't attempt). A statement
    inside an if/else branch's own body being unreachable doesn't make
    anything after the whole if/else statement unreachable -- only
    checking within one block at a time keeps that correct for free.
    """
    dead_lines: list[int] = []
    for node in ast.walk(tree):
        for block in _get_blocks(node):
            for i, stmt in enumerate(block[:-1]):
                if isinstance(stmt, _TERMINATING_STATEMENTS):
                    dead_lines.extend(s.lineno for s in block[i + 1 :])
                    break
    return dead_lines


# --- JavaScript advisory checks -------------------------------------
#
# tree-sitter only gives a raw syntax tree, not the clean statement-list
# abstraction ast.parse() gives Python -- so this is a real, separate
# undertaking, not a port. Two things make it structurally harder than
# the Python version, both handled by skipping the whole file rather
# than guessing (the same precedent as Python's star-import skip):
#
# 1. A single node type (ast.Name) with a ctx flag tells Python apart a
#    binding from a usage. JS's `identifier` node is used for BOTH, so
#    "is this a binding" has to be inferred from the PARENT node's type
#    and field, checked per binding shape (variable_declarator,
#    function/class names, parameters, imports, catch clauses).
# 2. Destructuring (`const {a, b} = obj`) binds names through a
#    different node type entirely (shorthand_property_identifier_pattern,
#    nested in object_pattern/array_pattern), not a plain identifier --
#    unlike Python, missing this wouldn't just under-flag, it would
#    actively false-positive (a real, correctly-bound destructured name
#    would look undefined). _js_has_unhandled_binding_shape() detects
#    any binding shape other than a plain identifier (or a plain
#    identifier with a simple default value) structurally -- so an
#    unrecognized future/rare shape is always treated as "skip", never
#    "silently mishandled".
_JS_LANGUAGE = Language(tree_sitter_javascript.language())
_JS_PARSER = Parser(_JS_LANGUAGE)

_JS_TERMINATING_STATEMENTS = frozenset(
    {"return_statement", "throw_statement", "break_statement", "continue_statement"}
)

# Hand-curated, deliberately generous across both browser and Node
# environments, since a given .js file's target runtime isn't known --
# unlike Python's dir(builtins), this is a heuristic list, not an
# authoritative one, and can't rule out missing an exotic global.
_JS_GLOBALS = frozenset(
    {
        # browser
        "window", "document", "console", "navigator", "location", "history",
        "localStorage", "sessionStorage", "fetch", "XMLHttpRequest", "FormData",
        "Headers", "Request", "Response", "URL", "URLSearchParams", "Event",
        "CustomEvent", "alert", "confirm", "prompt", "setTimeout", "setInterval",
        "clearTimeout", "clearInterval", "requestAnimationFrame",
        # node.js
        "require", "module", "exports", "process", "__dirname", "__filename",
        "global", "Buffer", "setImmediate",
        # language/runtime built-ins
        "Object", "Array", "Function", "String", "Number", "Boolean", "Symbol",
        "BigInt", "Math", "JSON", "Date", "RegExp", "Error", "TypeError",
        "RangeError", "SyntaxError", "ReferenceError", "EvalError", "URIError",
        "Promise", "Map", "Set", "WeakMap", "WeakSet", "Proxy", "Reflect",
        "ArrayBuffer", "DataView", "Int8Array", "Uint8Array", "Uint8ClampedArray",
        "Int16Array", "Uint16Array", "Int32Array", "Uint32Array", "Float32Array",
        "Float64Array", "BigInt64Array", "BigUint64Array", "Intl", "WebAssembly",
        "globalThis", "undefined", "NaN", "Infinity", "isNaN", "isFinite",
        "parseInt", "parseFloat", "encodeURIComponent", "decodeURIComponent",
        "encodeURI", "decodeURI", "structuredClone", "queueMicrotask", "arguments",
    }
)


def _walk_ts(node):
    yield node
    for child in node.children:
        yield from _walk_ts(child)


def _js_has_unhandled_binding_shape(root) -> bool:
    for node in _walk_ts(root):
        if node.type == "variable_declarator":
            name_node = node.child_by_field_name("name")
            if name_node is not None and name_node.type != "identifier":
                return True
        elif node.type == "formal_parameters":
            for child in node.children:
                if child.type in ("identifier", "(", ")", ","):
                    continue
                if child.type == "assignment_pattern":
                    left = child.child_by_field_name("left")
                    if left is not None and left.type == "identifier":
                        continue
                return True
        elif node.type == "catch_clause":
            param = node.child_by_field_name("parameter")
            if param is not None and param.type != "identifier":
                return True
        elif node.type == "arrow_function":
            param = node.child_by_field_name("parameter")
            if param is not None and param.type != "identifier":
                return True
    return False


def _collect_js_bound_names(root) -> set[str]:
    bound: set[str] = set()
    for node in _walk_ts(root):
        if node.type == "variable_declarator":
            name_node = node.child_by_field_name("name")
            if name_node is not None and name_node.type == "identifier":
                bound.add(name_node.text.decode("utf-8"))
        elif node.type in ("function_declaration", "generator_function_declaration", "class_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                bound.add(name_node.text.decode("utf-8"))
        elif node.type == "formal_parameters":
            for child in node.children:
                if child.type == "identifier":
                    bound.add(child.text.decode("utf-8"))
                elif child.type == "assignment_pattern":
                    left = child.child_by_field_name("left")
                    if left is not None and left.type == "identifier":
                        bound.add(left.text.decode("utf-8"))
        elif node.type == "arrow_function":
            param = node.child_by_field_name("parameter")
            if param is not None and param.type == "identifier":
                bound.add(param.text.decode("utf-8"))
        elif node.type == "import_specifier":
            alias = node.child_by_field_name("alias")
            target = alias if alias is not None else node.child_by_field_name("name")
            if target is not None and target.type == "identifier":
                bound.add(target.text.decode("utf-8"))
        elif node.type in ("namespace_import", "import_clause"):
            for child in node.children:
                if child.type == "identifier":
                    bound.add(child.text.decode("utf-8"))
        elif node.type == "catch_clause":
            param = node.child_by_field_name("parameter")
            if param is not None and param.type == "identifier":
                bound.add(param.text.decode("utf-8"))
    return bound


def _is_js_binding_occurrence(node) -> bool:
    """True if this specific identifier node IS a binding target itself
    (a declaration/parameter/import name), not a use of one -- so a
    declaration's own name is never flagged as reading an undefined
    variable."""
    parent = node.parent
    if parent is None:
        return False
    if parent.type == "variable_declarator":
        return parent.child_by_field_name("name") == node
    if parent.type in ("function_declaration", "generator_function_declaration", "class_declaration"):
        return parent.child_by_field_name("name") == node
    if parent.type == "formal_parameters":
        return True
    if parent.type == "assignment_pattern":
        return parent.child_by_field_name("left") == node
    if parent.type == "arrow_function":
        return parent.child_by_field_name("parameter") == node
    if parent.type == "import_specifier":
        # Both the external 'name' and local 'alias' (when renamed) are
        # excluded here -- 'name' isn't real local code at all once
        # aliased, and the actual local binding (alias, or name if there
        # is no alias) is already in the bound set.
        return True
    if parent.type in ("namespace_import", "import_clause"):
        return True
    if parent.type == "catch_clause":
        return parent.child_by_field_name("parameter") == node
    return False


def _find_js_undefined_names(root) -> list[tuple[str, int]]:
    if _js_has_unhandled_binding_shape(root):
        return []
    bound = _collect_js_bound_names(root)
    seen: set[str] = set()
    undefined: list[tuple[str, int]] = []
    for node in _walk_ts(root):
        if node.type != "identifier" or _is_js_binding_occurrence(node):
            continue
        name = node.text.decode("utf-8")
        if name not in bound and name not in _JS_GLOBALS and name not in seen:
            seen.add(name)
            undefined.append((name, node.start_point[0] + 1))
    return undefined


def _find_js_dead_code(root) -> list[int]:
    """Same deliberately narrow rule as Python's _find_dead_code(): only
    literal same-block sequential unreachability. Every JS block
    (function/if/else/for/while/try/catch bodies) is wrapped in an
    explicit statement_block node, so scanning each one independently,
    the same way Python scans each body/orelse/finalbody list
    independently, keeps an if/else's separate branches from cross-
    contaminating for free -- verified directly against real parsed
    if/else output, not assumed. switch/case bodies are deliberately not
    handled (a different block shape, not yet verified) -- under-
    covered, not mishandled.
    """
    dead_lines: list[int] = []
    for block_node in _walk_ts(root):
        if block_node.type != "statement_block":
            continue
        statements = [c for c in block_node.children if c.type not in ("{", "}")]
        for i, stmt in enumerate(statements[:-1]):
            if stmt.type in _JS_TERMINATING_STATEMENTS:
                dead_lines.extend(s.start_point[0] + 1 for s in statements[i + 1 :])
                break
    return dead_lines


# ----------------------------------------------------------------------

# Shared with edit_file.py/create_file.py: the exact separator used to
# append advisory_notes() output to a success message, and to detect
# afterward whether a given result string actually carries one (see
# Tool.should_show_result -- a single source of truth instead of the
# literal repeated at each call site, which could silently drift apart).
NOTE_MARKER = "\nNote: "


def _python_advisory_notes(content: str) -> list[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        # check_syntax() already validates this separately and raises
        # first in the real edit_file/create_file flow -- nothing
        # meaningful to analyze if we're ever called with invalid
        # content anyway.
        return []

    notes = [
        f"line {lineno}: '{name}' is used but doesn't appear to be "
        "imported or defined anywhere in this file -- this may cause a "
        "NameError at runtime."
        for name, lineno in _find_undefined_names(tree)
    ]
    dead_lines = _find_dead_code(tree)
    if dead_lines:
        notes.append(
            f"line {min(dead_lines)}: unreachable code -- follows an "
            "unconditional return/raise/break/continue earlier in the "
            "same block."
        )
    return notes


def _js_advisory_notes(content: str) -> list[str]:
    tree = _JS_PARSER.parse(content.encode("utf-8"))
    if tree.root_node.has_error:
        # check_syntax() already validates this separately and raises
        # first in the real edit_file/create_file flow.
        return []
    root = tree.root_node

    notes = [
        f"line {lineno}: '{name}' is used but doesn't appear to be "
        "imported or defined anywhere in this file -- this may cause a "
        "ReferenceError at runtime."
        for name, lineno in _find_js_undefined_names(root)
    ]
    dead_lines = _find_js_dead_code(root)
    if dead_lines:
        notes.append(
            f"line {min(dead_lines)}: unreachable code -- follows an "
            "unconditional return/throw/break/continue earlier in the "
            "same block."
        )
    return notes


def advisory_notes(path: Path, content: str) -> list[str]:
    """Non-blocking notes about content that's about to be written.

    Unlike check_syntax(), never prevents the write -- these are best-
    effort heuristic signals (an undefined name might come from a star
    import; even without one, static analysis can never rule out a name
    created via exec()/globals() at runtime, a real, permanent blind
    spot shared by every static tool, not a gap specific to this one)
    rather than the certainty a real syntax error is. Meant to be
    appended to a successful write's own result message so the model
    sees a real, specific, mechanically-derived fact -- not raised as an
    exception, and not a reason to block anything.

    Python and JavaScript (.js) only. Not .jsx -- JSX introduces node
    shapes (element/component names) never verified against this logic,
    so it's deliberately left unhandled rather than silently assumed
    safe. Not CSS/HTML -- neither "undefined name" nor "dead code"
    meaningfully applies to either (CSS has no variables in this sense
    beyond custom properties, a different and narrower check; HTML has
    no variables or control flow at all).
    """
    if path.suffix == ".py":
        return _python_advisory_notes(content)
    if path.suffix == ".js":
        return _js_advisory_notes(content)
    return []
