from agent_controller.tool_router import route_tools


def _schema(name: str, description: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": description}}


def test_returns_all_when_schema_count_at_min_keep():
    schemas = [_schema(f"tool_{i}", f"description {i}") for i in range(4)]
    result = route_tools("completely unrelated query text", schemas, min_keep=4)
    assert result == schemas


def test_returns_all_for_empty_query():
    schemas = [_schema(f"tool_{i}", f"description {i}") for i in range(6)]
    result = route_tools("", schemas, min_keep=4)
    assert result == schemas


def test_fails_open_when_no_keyword_overlap():
    schemas = [
        _schema("find_callers", "Find where a function is called."),
        _schema("read_file", "Read a file's contents."),
        _schema("list_directory", "List files in a directory."),
        _schema("run_command", "Run a shell command."),
        _schema("edit_file", "Edit a file with search and replace."),
    ]
    result = route_tools("zzz qqq xyz veryunusualword", schemas, min_keep=4)
    assert result == schemas


def test_narrows_and_keeps_matching_tool_when_signal_exists():
    schemas = [
        _schema("find_callers", "Find where a function is called."),
        _schema("find_importers", "Find which files import a module."),
        _schema("read_file", "Read a file's contents."),
        _schema("list_directory", "List files in a directory."),
        _schema("run_command", "Run a shell command."),
        _schema("edit_file", "Edit a file with search and replace."),
    ]
    result = route_tools("Where is the function foo called", schemas, min_keep=4)

    names = {s["function"]["name"] for s in result}
    assert "find_callers" in names
    assert len(result) < len(schemas)
    assert len(result) == 4


def test_never_drops_below_min_keep():
    schemas = [_schema(f"tool_{i}", f"description number {i} lorem ipsum") for i in range(8)]
    schemas.append(_schema("find_callers", "Find where a function is called."))

    result = route_tools("Where is the function foo called", schemas, min_keep=3)

    assert len(result) >= 3


def test_always_keep_names_survives_zero_score():
    """Reproduces the live regression this router caused: a request about
    'fixing a bug' shares no vocabulary at all with edit_file's name or
    description, while 'find'/'run' strongly matched three other tools by
    name — edit_file scored zero and was cut below the top-min_keep line,
    even though the task structurally required it. A caller-supplied
    always_keep_names must survive regardless of score."""
    schemas = [
        _schema("find_callers", "Find where a function is called."),
        _schema("find_importers", "Find which files import a module."),
        _schema("find_symbol", "Find where a class or function is defined."),
        _schema("run_command", "Run a shell command."),
        _schema("edit_file", "Create a new file or make a search/replace edit."),
    ]
    query = "There's a bug in clamp -- find and fix it, then run the tests"

    without_protection = route_tools(query, schemas, min_keep=4)
    assert "edit_file" not in {s["function"]["name"] for s in without_protection}

    protected = route_tools(query, schemas, min_keep=4, always_keep_names=frozenset({"edit_file"}))
    assert "edit_file" in {s["function"]["name"] for s in protected}


def test_fuzzy_prefix_matching_catches_word_variants():
    """Reproduces the exact live failure this router was built to fix:
    the query says 'called' but the tool is named 'find_callers' — a
    different word form of the same root, which exact-match keyword
    overlap alone would have missed."""
    schemas = [
        _schema("find_callers", "Find where a function or method is actually called."),
        _schema("read_file", "Read a file's contents."),
        _schema("list_directory", "List files in a directory."),
        _schema("run_command", "Run a shell command."),
        _schema("edit_file", "Edit a file with search and replace."),
        _schema("repo_overview", "Get a structural overview of the project."),
    ]

    result = route_tools("Where is the function cap_observation actually called?", schemas, min_keep=4)

    names = {s["function"]["name"] for s in result}
    assert "find_callers" in names
