from agent_controller.tool_router import (
    ToolEmbeddingCache,
    _fractional_ranks,
    route_tools,
    route_tools_hybrid,
)
from model_interface.base import ModelInterface, ModelResponse, ModelUnavailableError


def test_fractional_ranks_assigns_1_to_the_highest_score():
    ranks = _fractional_ranks({"a": 10, "b": 5, "c": 1})
    assert ranks == {"a": 1, "b": 2, "c": 3}


def test_fractional_ranks_averages_a_tied_group():
    # b and c tie for 2nd/3rd place -> both get rank 2.5, d correctly
    # moves to rank 4 rather than 3 (a tie doesn't shrink the ranking).
    ranks = _fractional_ranks({"a": 10, "b": 5, "c": 5, "d": 1})
    assert ranks == {"a": 1, "b": 2.5, "c": 2.5, "d": 4}


def test_fractional_ranks_handles_every_score_tied():
    ranks = _fractional_ranks({"a": 1, "b": 1, "c": 1})
    assert ranks == {"a": 2, "b": 2, "c": 2}


def _schema(name: str, description: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {}}}


class ControlledEmbedModel(ModelInterface):
    """Returns an exact, pre-assigned vector per text (keyed by the exact
    string passed to embed()), so a test can construct precise similarity
    scenarios instead of relying on a real or word-overlap-approximated
    embedding. raise_for lets a specific text simulate a real embedding
    failure (network hiccup, etc.) instead of returning a vector."""

    def __init__(self, vectors: dict[str, list[float]], raise_for: frozenset[str] = frozenset()):
        self._vectors = vectors
        self._raise_for = raise_for
        self.embed_calls: list[str] = []

    def generate(self, messages, tools=None):
        return ModelResponse(text="")

    def embed(self, text):
        self.embed_calls.append(text)
        if text in self._raise_for:
            raise ModelUnavailableError("embedding service unavailable")
        return self._vectors[text]


_EDIT_FILE = _schema("edit_file", "create search replace")
_FIND_SYMBOL = _schema("find_symbol", "find symbol definition")
_FIND_CALLERS = _schema("find_callers", "find callers of a function")
_FIND_IMPORTERS = _schema("find_importers", "find importers of a module")
_FIND_HTML = _schema("find_html", "find html script and stylesheet references")
_RUN_COMMAND = _schema("run_command", "run a shell command")
_LIST_DIRECTORY = _schema("list_directory", "list files in a directory")
# Four "find_*" tools deliberately -- with _DEFAULT_MIN_KEEP == 4, this
# means the min-keep padding fallback never needs to reach into the
# zero-scored group, so edit_file/run_command/list_directory are
# unambiguously excluded under keyword-only routing, not just
# coincidentally excluded by list order.
_ALL_SCHEMAS = [
    _EDIT_FILE, _FIND_SYMBOL, _FIND_CALLERS, _FIND_IMPORTERS, _FIND_HTML, _RUN_COMMAND, _LIST_DIRECTORY,
]

_QUERY = "there's a bug, find and fix it"


def _tool_text(schema: dict) -> str:
    fn = schema["function"]
    return f"{fn['name']}: {fn['description']}"


def test_promotes_a_conceptually_relevant_tool_that_keyword_scoring_would_bury():
    """Reproduces the real, documented edit_file/"there's a bug, fix it"
    gap: keyword scoring alone gives edit_file a score of 0 (its
    description shares no vocabulary with the request), while
    find_symbol/find_callers/find_importers all score highly on "find"
    alone. A real embedding check (done live against this project's
    actual tools) ranked edit_file as highly relevant despite that --
    this reproduces the same shape with controlled, exact vectors."""
    vectors = {
        _QUERY: [1.0, 0.0],
        _tool_text(_EDIT_FILE): [1.0, 0.0],  # identical direction -> similarity 1.0
        _tool_text(_FIND_SYMBOL): [0.0, 1.0],
        _tool_text(_FIND_CALLERS): [0.0, 1.0],
        _tool_text(_FIND_IMPORTERS): [0.0, 1.0],
        _tool_text(_FIND_HTML): [0.0, 1.0],
        _tool_text(_RUN_COMMAND): [0.0, 1.0],
        _tool_text(_LIST_DIRECTORY): [0.0, 1.0],
    }
    model = ControlledEmbedModel(vectors)
    cache = ToolEmbeddingCache(model)

    # Sanity check first: under keyword-only routing, edit_file really is
    # excluded -- otherwise this test wouldn't be proving anything.
    keyword_only = route_tools(_QUERY, _ALL_SCHEMAS)
    assert "edit_file" not in {s["function"]["name"] for s in keyword_only}

    hybrid = route_tools_hybrid(_QUERY, _ALL_SCHEMAS, cache)
    names = {s["function"]["name"] for s in hybrid}
    assert "edit_file" in names
    # The clearly irrelevant tools (zero on both signals) should still
    # be excluded -- this isn't just "keep everything".
    assert "run_command" not in names
    assert "list_directory" not in names


def test_falls_back_to_keyword_only_routing_when_the_query_embedding_fails():
    """A network hiccup on the query embedding must never make routing
    worse than today's keyword-only baseline -- delegates to route_tools()
    itself (not a reimplementation), so behavior is guaranteed identical,
    not just similar."""
    model = ControlledEmbedModel(vectors={}, raise_for=frozenset({_QUERY}))
    cache = ToolEmbeddingCache(model)

    hybrid_result = route_tools_hybrid(_QUERY, _ALL_SCHEMAS, cache)
    keyword_result = route_tools(_QUERY, _ALL_SCHEMAS)

    assert [s["function"]["name"] for s in hybrid_result] == [s["function"]["name"] for s in keyword_result]


def test_a_single_tools_embedding_failure_does_not_crash_the_whole_step():
    """One tool's embedding call failing (not the query's) should only
    zero out that tool's embedding signal, not blow up the entire
    routing decision for every other tool."""
    vectors = {
        _QUERY: [1.0, 0.0],
        _tool_text(_EDIT_FILE): [1.0, 0.0],
        _tool_text(_FIND_SYMBOL): [0.0, 1.0],
        _tool_text(_FIND_CALLERS): [0.0, 1.0],
        _tool_text(_FIND_IMPORTERS): [0.0, 1.0],
        _tool_text(_FIND_HTML): [0.0, 1.0],
        _tool_text(_RUN_COMMAND): [0.0, 1.0],
        _tool_text(_LIST_DIRECTORY): [0.0, 1.0],
    }
    model = ControlledEmbedModel(vectors, raise_for=frozenset({_tool_text(_FIND_SYMBOL)}))
    cache = ToolEmbeddingCache(model)

    result = route_tools_hybrid(_QUERY, _ALL_SCHEMAS, cache)

    # Must not raise, and edit_file (unaffected by the failure) should
    # still be correctly promoted.
    assert "edit_file" in {s["function"]["name"] for s in result}


def test_always_keep_names_survives_regardless_of_combined_score():
    vectors = {
        _QUERY: [1.0, 0.0],
        _tool_text(_EDIT_FILE): [0.0, 1.0],
        _tool_text(_FIND_SYMBOL): [0.0, 1.0],
        _tool_text(_FIND_CALLERS): [0.0, 1.0],
        _tool_text(_FIND_IMPORTERS): [0.0, 1.0],
        _tool_text(_FIND_HTML): [0.0, 1.0],
        _tool_text(_RUN_COMMAND): [0.0, 1.0],
        _tool_text(_LIST_DIRECTORY): [0.0, 1.0],
    }
    model = ControlledEmbedModel(vectors)
    cache = ToolEmbeddingCache(model)

    result = route_tools_hybrid(
        "totally unrelated wording", _ALL_SCHEMAS, cache, always_keep_names=frozenset({"list_directory"})
    )

    assert "list_directory" in {s["function"]["name"] for s in result}


def test_tool_embedding_cache_only_embeds_each_tool_once():
    model = ControlledEmbedModel({_tool_text(_EDIT_FILE): [1.0, 0.0]})
    cache = ToolEmbeddingCache(model)

    cache.tool_vector(_EDIT_FILE)
    cache.tool_vector(_EDIT_FILE)
    cache.tool_vector(_EDIT_FILE)

    assert model.embed_calls == [_tool_text(_EDIT_FILE)]


def test_tool_embedding_cache_query_vector_is_not_cached():
    model = ControlledEmbedModel({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    cache = ToolEmbeddingCache(model)

    cache.query_vector("a")
    cache.query_vector("b")

    assert model.embed_calls == ["a", "b"]


def test_returns_none_when_a_tools_embedding_fails():
    model = ControlledEmbedModel({}, raise_for=frozenset({_tool_text(_EDIT_FILE)}))
    cache = ToolEmbeddingCache(model)

    assert cache.tool_vector(_EDIT_FILE) is None


def test_returns_none_when_the_query_embedding_fails():
    model = ControlledEmbedModel({}, raise_for=frozenset({_QUERY}))
    cache = ToolEmbeddingCache(model)

    assert cache.query_vector(_QUERY) is None
