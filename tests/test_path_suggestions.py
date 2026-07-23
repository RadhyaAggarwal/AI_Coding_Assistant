from pathlib import Path

from tools.path_suggestions import suggest_similar_directories, suggest_similar_paths


def test_suggests_the_real_live_near_miss(tmp_path):
    """Reproduces the exact live gap: the model asked for
    'templates/notes.html' when the real file was
    'core/templates/core/note_list.html'."""
    real = tmp_path / "core" / "templates" / "core" / "note_list.html"
    real.parent.mkdir(parents=True)
    real.write_text("<p>hi</p>", encoding="utf-8")

    suggestions = suggest_similar_paths(tmp_path, "templates/notes.html")

    assert "core/templates/core/note_list.html" in suggestions


def test_returns_empty_when_no_file_with_that_extension_exists(tmp_path):
    (tmp_path / "views.py").write_text("x = 1\n", encoding="utf-8")
    assert suggest_similar_paths(tmp_path, "missing.html") == []


def test_returns_empty_for_a_path_with_no_extension(tmp_path):
    (tmp_path / "README").write_text("hi", encoding="utf-8")
    assert suggest_similar_paths(tmp_path, "readme") == []


def test_falls_back_to_listing_when_nothing_is_a_close_match(tmp_path):
    """A wildly wrong guess still gets real candidates back, not an
    empty list -- better to show something real than nothing at all."""
    (tmp_path / "note_list.html").write_text("<p>hi</p>", encoding="utf-8")
    (tmp_path / "note_form.html").write_text("<p>hi</p>", encoding="utf-8")

    suggestions = suggest_similar_paths(tmp_path, "zzz_completely_unrelated.html")

    assert set(suggestions) == {"note_list.html", "note_form.html"}


def test_skips_conventional_build_and_cache_directories(tmp_path):
    """Reuses repo_index.scanner's SKIP_DIR_NAMES -- a file sitting in
    node_modules or __pycache__ is never a real suggestion."""
    skipped = tmp_path / "node_modules" / "notes.html"
    skipped.parent.mkdir(parents=True)
    skipped.write_text("junk", encoding="utf-8")

    assert suggest_similar_paths(tmp_path, "notes.html") == []


def test_caps_the_number_of_suggestions(tmp_path):
    for i in range(10):
        (tmp_path / f"note_{i}.html").write_text("hi", encoding="utf-8")

    suggestions = suggest_similar_paths(tmp_path, "note_x.html")

    assert len(suggestions) <= 5


def test_uses_forward_slashes(tmp_path):
    nested = tmp_path / "core" / "templates" / "note_list.html"
    nested.parent.mkdir(parents=True)
    nested.write_text("hi", encoding="utf-8")

    suggestions = suggest_similar_paths(tmp_path, "note_list.html")

    assert all("\\" not in s for s in suggestions)


def test_suggests_a_similarly_named_real_directory(tmp_path):
    (tmp_path / "core" / "templates").mkdir(parents=True)

    suggestions = suggest_similar_directories(tmp_path, "template")

    assert "core/templates" in suggestions


def test_directory_suggestions_returns_empty_when_no_directories_exist(tmp_path):
    (tmp_path / "views.py").write_text("x = 1\n", encoding="utf-8")
    assert suggest_similar_directories(tmp_path, "templates") == []


def test_directory_suggestions_skips_conventional_build_and_cache_directories(tmp_path):
    (tmp_path / "node_modules" / "templates").mkdir(parents=True)
    assert suggest_similar_directories(tmp_path, "templates") == []


def test_directory_suggestions_falls_back_when_nothing_is_a_close_match(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "mysite").mkdir()

    suggestions = suggest_similar_directories(tmp_path, "zzz_completely_unrelated")

    assert set(suggestions) == {"core", "mysite"}


def test_directory_suggestions_point_at_the_real_containing_directory_of_a_file(tmp_path):
    """Reproduces the exact live misuse: search_code's 'path' argument
    means "directory to search within", and a model passed a real FILE
    path (core/views.py) instead. Its actual containing directory is a
    certain fact, not a guess -- must be returned directly, not buried
    behind fuzzy name matching against unrelated real directories (which
    for a name like "views.py" would find nothing close anyway)."""
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "views.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "deploy").mkdir()
    (tmp_path / "tests").mkdir()

    suggestions = suggest_similar_directories(tmp_path, "core/views.py")

    assert suggestions == ["core"]


def test_directory_suggestions_for_a_file_at_the_project_root(tmp_path):
    (tmp_path / "manage.py").write_text("x = 1\n", encoding="utf-8")

    suggestions = suggest_similar_directories(tmp_path, "manage.py")

    assert suggestions == ["."]
