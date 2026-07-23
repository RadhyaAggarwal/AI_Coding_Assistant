from pathlib import Path

import pytest

from tools.syntax_check import InvalidSyntaxError, advisory_notes, check_syntax


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


def test_rejection_hints_at_template_tag_quote_conflict_for_html():
    """Reproduces the exact live bug: a model wrote
    href='{% url 'delete_note' note.id %}' -- valid, idiomatic Django
    template syntax, but tree-sitter's HTML grammar doesn't understand
    {% %} and sees the tag's own quote as closing the attribute early.
    The rejection itself must still happen (this never changes what's
    accepted), but the message should point at the real, likely cause."""
    broken = "<a href='{% url 'delete_note' note.id %}'>Delete</a>"
    with pytest.raises(InvalidSyntaxError, match="template-tag syntax"):
        check_syntax(Path("x.html"), broken)


def test_rejection_hints_at_template_tag_quote_conflict_for_js():
    broken = "var apiUrl = '{% url 'api_endpoint' %}';"
    with pytest.raises(InvalidSyntaxError, match="template-tag syntax"):
        check_syntax(Path("x.js"), broken)


def test_rejection_hints_at_template_tag_quote_conflict_for_css():
    broken = "content: \"{% trans 'Some String' %}\";"
    with pytest.raises(InvalidSyntaxError, match="template-tag syntax"):
        check_syntax(Path("x.css"), broken)


def test_rejection_stays_generic_when_no_template_tags_present():
    """The hint must not appear on an ordinary, unrelated syntax error --
    it's specific to content that actually contains {% %}/{{ }}."""
    with pytest.raises(InvalidSyntaxError) as exc_info:
        check_syntax(Path("x.html"), '<div id="x"')
    assert "template-tag syntax" not in str(exc_info.value)


def test_rejection_stays_generic_for_python_even_with_template_tag_text():
    """Scoped to the tree-sitter-checked languages only -- a Python
    SyntaxError is a different, unrelated failure shape, even if the
    broken content happens to contain {% %}-looking text."""
    with pytest.raises(InvalidSyntaxError) as exc_info:
        check_syntax(Path("x.py"), "def f(:\n    x = '{% not python %}'\n")
    assert "template-tag syntax" not in str(exc_info.value)


def test_advisory_notes_catches_the_real_live_missing_import():
    """Reproduces the exact live bug: core/views.py's delete_note used
    get_object_or_404 without ever importing it -- a real NameError at
    runtime, invisible to check_syntax() (compile() can't catch this,
    see the module docstring on advisory_notes for why), and invisible
    to manage.py check too. Advisory, not blocking -- see
    test_advisory_notes_never_raises."""
    content = (
        "from django.shortcuts import render, redirect\n"
        "from .models import Note\n\n"
        "def delete_note(request, note_id):\n"
        "    note = get_object_or_404(Note, id=note_id)\n"
        "    note.delete()\n"
        "    return redirect('note_count')\n"
    )
    notes = advisory_notes(Path("core/views.py"), content)
    assert len(notes) == 1
    assert "get_object_or_404" in notes[0]
    assert "line 5" in notes[0]


def test_advisory_notes_catches_the_real_live_dead_code():
    """Reproduces the exact live bug: delete_note's edit left a second,
    unreachable return statement after the real one, from re-typing
    index's body as part of an insertion instead of leaving the
    original, never-removed line alone."""
    content = (
        "def delete_note(request, note_id):\n"
        "    note = note_id\n"
        "    return note\n"
        "    return None\n"
    )
    notes = advisory_notes(Path("core/views.py"), content)
    assert len(notes) == 1
    assert "unreachable" in notes[0]
    assert "line 4" in notes[0]


def test_advisory_notes_does_not_flag_names_actually_defined():
    """The common case must stay quiet -- imports, parameters, local
    assignments, and builtins should never be flagged."""
    content = (
        "import os\n"
        "from django.shortcuts import render\n\n"
        "def f(request, note_id):\n"
        "    path = os.path.join('a', 'b')\n"
        "    total = len([1, 2, 3])\n"
        "    return render(request, path, {'total': total, 'note_id': note_id})\n"
    )
    assert advisory_notes(Path("x.py"), content) == []


def test_advisory_notes_does_not_flag_a_conditionally_imported_name():
    """A name imported in one branch of a try/except (a genuinely common,
    idiomatic pattern -- e.g. preferring a fast JSON library, falling
    back to the stdlib one) must not be flagged just because a static
    walk can't know which branch actually runs at runtime -- it only
    needs to see the name bound *somewhere* in the file."""
    content = "try:\n    import ujson as json\nexcept ImportError:\n    import json\n\njson.dumps({})\n"
    assert advisory_notes(Path("x.py"), content) == []


def test_advisory_notes_stays_silent_on_a_star_import_file():
    """The real risk this project's own recalibration surfaced: a star
    import (from abc import *, or Django's own idiomatic
    settings/production.py doing `from .base import *`) binds an unknown
    set of names a static walk can't see into. Rather than guess and
    risk being confidently wrong on real, valid code, the whole file is
    left unchecked -- proven here with a name (abstractmethod) that
    really does come from the star import and must not be flagged."""
    content = "from abc import *\n\nclass Base:\n    @abstractmethod\n    def do_thing(self):\n        pass\n"
    assert advisory_notes(Path("x.py"), content) == []


def test_advisory_notes_does_not_flag_a_genuinely_undefined_name_in_a_star_import_file():
    """The direct cost of the star-import carve-out, made explicit: once
    a file has a star import, coverage is lost for the *whole* file, not
    just names plausibly from the star-imported module -- a real,
    disclosed tradeoff, not a hidden one."""
    content = "from abc import *\n\ngenuinely_undefined_name()\n"
    assert advisory_notes(Path("x.py"), content) == []


def test_advisory_notes_does_not_flag_unreachable_looking_code_in_a_different_if_branch():
    """Deliberately narrow scope, proven: a return inside one if-branch
    must not make code after the whole if/else statement look dead --
    that would require real "does every branch return" control-flow
    analysis, which this intentionally does not attempt. Only literal
    same-block sequential unreachability is flagged."""
    content = "def f(x):\n    if x:\n        return 1\n    return 2\n"
    assert advisory_notes(Path("x.py"), content) == []


def test_advisory_notes_does_not_flag_match_case_capture_patterns():
    """Reproduces a real false positive found by a post-session coherence
    audit: MatchAs/MatchStar/MatchMapping bind a captured name as a
    plain string attribute on the pattern node, not an ast.Name node --
    invisible to the ast.Name/Store walk the undefined-name check
    otherwise relies on, so a genuinely valid capture like 'direction'
    or 'rest' below was wrongly flagged as undefined before this was
    fixed."""
    content = (
        "def handle(command):\n"
        "    match command:\n"
        "        case ['go', direction]:\n"
        "            print(direction)\n"
        "        case {'x': 0, 'y': 0, **rest}:\n"
        "            print(rest)\n"
        "        case other:\n"
        "            print(other)\n"
    )
    assert advisory_notes(Path("x.py"), content) == []


def test_advisory_notes_does_not_flag_unreachable_looking_code_in_a_different_match_case():
    """Same requirement as the existing if/else test, for match/case: a
    return inside one case's body must not make code after the whole
    match statement look dead. Unlike an exhaustive match, this one has
    no wildcard case, so the trailing return is genuinely reachable
    (x != 1 falls through) -- a real correctness requirement, not just
    the deliberately-tolerated under-flagging of a truly dead line."""
    content = (
        "def f(x):\n"
        "    match x:\n"
        "        case 1:\n"
        "            return 'one'\n"
        "    return 'other'\n"
    )
    assert advisory_notes(Path("x.py"), content) == []


def test_advisory_notes_still_catches_dead_code_within_one_match_case():
    content = (
        "def f(x):\n"
        "    match x:\n"
        "        case 1:\n"
        "            return 'one'\n"
        "            print('dead')\n"
        "        case _:\n"
        "            return 'other'\n"
    )
    notes = advisory_notes(Path("x.py"), content)
    assert len(notes) == 1
    assert "unreachable" in notes[0]
    assert "line 5" in notes[0]


def test_advisory_notes_does_not_flag_code_in_a_finally_block_after_a_try_return():
    """A return inside try.body must not make code in finally.body look
    dead -- finally always runs regardless of whether try returned."""
    content = "def f():\n    def cleanup():\n        pass\n    try:\n        return 1\n    finally:\n        cleanup()\n"
    assert advisory_notes(Path("x.py"), content) == []


def test_advisory_notes_only_checks_python_files():
    assert advisory_notes(Path("x.js"), "undefined_name();") == []
    assert advisory_notes(Path("x.html"), "<p>undefined_name</p>") == []


def test_advisory_notes_returns_empty_on_invalid_python_rather_than_raising():
    """check_syntax() already validates and raises first in the real
    edit_file/create_file flow -- if this is ever called with invalid
    content anyway, there's nothing meaningful to analyze, and it must
    never itself become a new way to crash."""
    assert advisory_notes(Path("x.py"), "def f(:\n") == []


def test_advisory_notes_never_raises_only_check_syntax_does():
    """The core design distinction this whole mechanism rests on: a real
    syntax error is a certainty (compile() doesn't guess), so
    check_syntax() blocks the write. An undefined name or dead code is a
    best-effort heuristic (a star import, or a name created via exec(),
    could make it wrong) -- advisory_notes() must never raise, only
    return notes to attach to an already-successful write."""
    content = "def f():\n    return get_object_or_404_typo(1)\n"
    check_syntax(Path("x.py"), content)  # does not raise -- this is valid Python
    notes = advisory_notes(Path("x.py"), content)
    assert len(notes) == 1  # still surfaced, just not as a blocking error
