import json
import os
import time

from repo_index.cache import cache_path_for
from repo_index.indexer import RepoIndex


def test_reuses_cached_file_index_when_mtime_and_size_unchanged(tmp_path):
    """Proves the cache is genuinely consulted, not just incidentally
    correct: overwrite the file with different (but same-length) content
    while forcing back its original mtime, then check the STALE cached
    result is what comes back."""
    file_path = tmp_path / "a.py"
    original = "def foo():\n    pass\n"
    file_path.write_text(original, encoding="utf-8")
    stat_before = file_path.stat()

    index1 = RepoIndex(tmp_path)
    assert len(index1.find_symbol("foo")) == 1

    replacement = "def bar():\n    pass\n"
    assert len(replacement) == len(original)
    file_path.write_text(replacement, encoding="utf-8")
    os.utime(file_path, (stat_before.st_atime, stat_before.st_mtime))

    index2 = RepoIndex(tmp_path)
    assert len(index2.find_symbol("foo")) == 1
    assert len(index2.find_symbol("bar")) == 0


def test_reparses_when_mtime_changes(tmp_path):
    file_path = tmp_path / "a.py"
    file_path.write_text("def foo():\n    pass\n", encoding="utf-8")
    RepoIndex(tmp_path)  # populate the cache

    time.sleep(0.05)  # ensure a distinctly newer mtime
    file_path.write_text("def bar():\n    pass\n", encoding="utf-8")

    index2 = RepoIndex(tmp_path)
    assert len(index2.find_symbol("bar")) == 1
    assert len(index2.find_symbol("foo")) == 0


def test_survives_corrupt_cache_file(tmp_path):
    file_path = tmp_path / "a.py"
    file_path.write_text("def foo():\n    pass\n", encoding="utf-8")
    cache_file = cache_path_for(tmp_path)
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("not valid json {{{", encoding="utf-8")

    index = RepoIndex(tmp_path)

    assert len(index.find_symbol("foo")) == 1


def test_deleted_file_dropped_from_cache(tmp_path):
    file_a = tmp_path / "a.py"
    file_b = tmp_path / "b.py"
    file_a.write_text("def foo():\n    pass\n", encoding="utf-8")
    file_b.write_text("def bar():\n    pass\n", encoding="utf-8")
    RepoIndex(tmp_path)

    file_b.unlink()
    RepoIndex(tmp_path)

    cache_data = json.loads(cache_path_for(tmp_path).read_text(encoding="utf-8"))
    assert "a.py" in cache_data
    assert "b.py" not in cache_data


def test_cache_file_written_after_build(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")

    RepoIndex(tmp_path)

    assert cache_path_for(tmp_path).is_file()
