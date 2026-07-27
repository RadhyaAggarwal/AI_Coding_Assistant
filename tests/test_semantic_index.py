import os
import time

import repo_index.semantic_index as semantic_index
from model_interface.base import ModelInterface, ModelResponse
from repo_index.semantic_index import SemanticIndex, _chunk_boundaries, _split_by_char_limit
from repo_index.models import Symbol


class WordVectorModel(ModelInterface):
    """Deterministic fake: embeds text as word-presence counts over a
    small fixed vocabulary, so semantically related text gets similar
    vectors -- enough to prove ranking behaves sensibly without needing
    a real embedding model."""

    VOCAB = ["email", "validate", "user", "delete", "note", "password", "send"]

    def __init__(self):
        self.embed_calls = 0

    def generate(self, messages, tools=None):
        return ModelResponse(text="")

    def embed(self, text):
        self.embed_calls += 1
        lower = text.lower()
        return [float(lower.count(word)) for word in self.VOCAB]


class CountingModel(ModelInterface):
    """Doesn't need meaningful vectors -- just counts real embed() calls,
    for proving caching behavior precisely."""

    def __init__(self):
        self.embed_calls = 0

    def generate(self, messages, tools=None):
        return ModelResponse(text="")

    def embed(self, text):
        self.embed_calls += 1
        return [float(len(text) % 7), float(text.count("e"))]


def test_chunk_boundaries_splits_on_the_next_symbols_start_line():
    symbols = [Symbol(name="foo", kind="function", file="a.py", line=1), Symbol(name="bar", kind="function", file="a.py", line=4)]
    boundaries = _chunk_boundaries(symbols, total_lines=6)
    assert boundaries == [("foo", 1, 3), ("bar", 4, 6)]


def test_chunk_boundaries_includes_a_module_header_before_the_first_symbol():
    symbols = [Symbol(name="foo", kind="function", file="a.py", line=3)]
    boundaries = _chunk_boundaries(symbols, total_lines=5)
    assert boundaries[0] == ("(module header)", 1, 2)
    assert boundaries[1] == ("foo", 3, 5)


def test_chunk_boundaries_falls_back_to_one_whole_file_chunk_with_no_symbols():
    assert _chunk_boundaries([], total_lines=10) == [("(whole file)", 1, 10)]


def test_chunk_boundaries_caps_a_very_long_trailing_symbol():
    symbols = [Symbol(name="huge", kind="function", file="a.py", line=1)]
    boundaries = _chunk_boundaries(symbols, total_lines=500)
    name, start, end = boundaries[0]
    assert (end - start + 1) <= 100


def test_ranks_the_semantically_relevant_chunk_highest(tmp_path):
    (tmp_path / "validators.py").write_text(
        "def validate_email(addr):\n"
        "    \"\"\"Check the email address looks valid.\"\"\"\n"
        "    return '@' in addr\n"
        "\n"
        "def send_password_reset(user):\n"
        "    email = user.email\n"
        "    return f'sent to {email}'\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.py").write_text(
        "def delete_note(note_id):\n"
        "    \"\"\"Remove a note from the database.\"\"\"\n"
        "    return note_id\n",
        encoding="utf-8",
    )

    index = SemanticIndex(tmp_path, WordVectorModel())
    results = index.search("how do we check if an email is valid", top_k=3)

    assert results[0][0].name == "validate_email"
    assert results[-1][0].name == "delete_note"
    assert results[0][1] > results[-1][1]


def test_returns_empty_list_for_a_project_with_no_indexable_files(tmp_path):
    index = SemanticIndex(tmp_path, WordVectorModel())
    assert index.search("anything") == []


def test_does_not_reembed_unchanged_files_on_a_second_build(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def baz():\n    return 3\n", encoding="utf-8")
    model = CountingModel()

    SemanticIndex(tmp_path, model)
    first_build_calls = model.embed_calls
    assert first_build_calls == 3  # foo, bar, baz

    SemanticIndex(tmp_path, model)
    assert model.embed_calls == first_build_calls  # nothing re-embedded


def test_only_reembeds_the_changed_files_chunks_after_an_edit(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def baz():\n    return 3\n", encoding="utf-8")
    model = CountingModel()

    SemanticIndex(tmp_path, model)
    assert model.embed_calls == 3

    time.sleep(0.02)
    (tmp_path / "a.py").write_text("def foo():\n    return 100\n\ndef bar():\n    return 2\n", encoding="utf-8")
    SemanticIndex(tmp_path, model)

    # Only a.py's 2 chunks (foo, bar) should be re-embedded -- b.py's
    # untouched baz chunk must not be recomputed.
    assert model.embed_calls == 5


def test_reuses_the_stale_cached_result_when_mtime_and_size_are_unchanged(tmp_path):
    """Same proof-of-genuine-caching pattern as
    test_repo_index_cache.py's equivalent test: overwrite with different
    (same-length) content but force back the original mtime, and confirm
    the STALE cached vector is what's actually reused."""
    file_path = tmp_path / "a.py"
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    stat_before = file_path.stat()
    model = CountingModel()

    SemanticIndex(tmp_path, model)
    assert model.embed_calls == 1

    file_path.write_text("def foo():\n    return 9\n", encoding="utf-8")  # same length
    os.utime(file_path, (stat_before.st_atime, stat_before.st_mtime))

    SemanticIndex(tmp_path, model)
    assert model.embed_calls == 1  # still using the stale cached vector


def test_on_progress_is_not_called_when_everything_is_already_cached(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    model = CountingModel()
    SemanticIndex(tmp_path, model)  # first build, populates the cache

    events = []
    SemanticIndex(tmp_path, model, on_progress=events.append)
    assert events == []


def test_split_by_char_limit_breaks_only_on_line_boundaries():
    lines = ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd"]  # 10 chars each
    assert _split_by_char_limit(lines, 1, 4, max_chars=25) == [(1, 2), (3, 4)]


def test_split_by_char_limit_keeps_an_oversized_single_line_as_its_own_range():
    lines = ["a" * 50]
    assert _split_by_char_limit(lines, 1, 1, max_chars=10) == [(1, 1)]


def test_oversized_chunk_is_split_into_multiple_embeddable_pieces(tmp_path, monkeypatch):
    """Live-discovered bug: a single large chunk (originally a 214-line,
    13,351-character module-header chunk in this project's own loop.py)
    was sent whole to a real embedding model and rejected as too large
    for its context window. A chunk over _MAX_CHUNK_CHARS must be split
    into embeddable pieces instead of failing or being truncated."""
    monkeypatch.setattr(semantic_index, "_MAX_CHUNK_CHARS", 40)
    (tmp_path / "a.py").write_text(
        "def big():\n"
        "    line_one_here\n"
        "    line_two_here\n"
        "    line_three_here\n"
        "    line_four_here\n"
        "    line_five_here\n",
        encoding="utf-8",
    )
    seen_texts = []

    class RecordingModel(ModelInterface):
        def generate(self, messages, tools=None):
            return ModelResponse(text="")

        def embed(self, text):
            seen_texts.append(text)
            return [1.0, 0.0]

    index = SemanticIndex(tmp_path, RecordingModel())

    assert len(seen_texts) > 1  # one chunk had to become several embed() calls
    assert all(len(t) <= 40 for t in seen_texts)
    names = [chunk.name for chunk, _ in index.search("big", top_k=10)]
    assert any("big (part 1/" in name for name in names)


def test_a_single_oversized_line_is_skipped_not_embedded(tmp_path, monkeypatch):
    """The one case splitting can't fix: one line, by itself, already
    over the limit (e.g. minified/generated content). Must be skipped,
    not truncated -- truncating would silently embed a corrupted,
    arbitrarily-cut fragment (a deliberate choice, not an oversight)."""
    monkeypatch.setattr(semantic_index, "_MAX_CHUNK_CHARS", 20)
    (tmp_path / "a.py").write_text(
        "def bar():\n"
        "    this_one_single_line_is_way_too_long_to_ever_fit_the_cap = 1\n",
        encoding="utf-8",
    )
    events = []
    index = SemanticIndex(tmp_path, CountingModel(), on_progress=events.append)

    assert any("Skipping" in e and "a.py" in e for e in events)
    results = index.search("bar", top_k=10)
    assert not any("too_long_to_ever_fit" in chunk.text for chunk, _ in results)


def test_skip_warning_is_not_repeated_on_an_unchanged_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic_index, "_MAX_CHUNK_CHARS", 20)
    (tmp_path / "a.py").write_text(
        "def bar():\n"
        "    this_one_single_line_is_way_too_long_to_ever_fit_the_cap = 1\n",
        encoding="utf-8",
    )
    model = CountingModel()

    first_events = []
    SemanticIndex(tmp_path, model, on_progress=first_events.append)
    assert any("Skipping" in e for e in first_events)

    second_events = []
    SemanticIndex(tmp_path, model, on_progress=second_events.append)
    assert not any("Skipping" in e for e in second_events)


def test_on_progress_reports_the_real_count_of_new_chunks(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n", encoding="utf-8")
    model = CountingModel()

    events = []
    SemanticIndex(tmp_path, model, on_progress=events.append)

    assert len(events) == 1
    assert "2" in events[0]
    assert "Embedding" in events[0]
