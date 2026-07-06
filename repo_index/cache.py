"""Disk-persisted cache for RepoIndex's per-file parse results, keyed by
each file's mtime and size.

Walking the project tree is cheap (just stat calls); re-parsing every
file via tree-sitter/ast on every single RepoIndex() build is not, and
that cost is paid on every call to find_symbol/find_importers/
find_callers — including several times within one request now that the
loop can chain multiple tool calls per turn, and again on every separate
CLI invocation even when nothing in the project changed. A file whose
(mtime, size) still match its cached entry is assumed unchanged and its
parsed FileIndex is reused rather than re-parsed.

The cache file only ever contains entries for files the current build
actually walked, so a deleted file's stale entry is dropped automatically
rather than accumulating forever. A missing, corrupt, or (after a future
model-shape change) incompatible cache is not worth failing over — it's
treated as empty and rebuilt from scratch.

Mtime+size alone only catches a file's *own* content changing — it has
no way to know a future change to indexing logic itself (a parser bug
fix, a language moving from unsupported to indexed, an extension-mapping
change) should invalidate everything, not just files that happen to
change afterward. _CACHE_VERSION covers that: bump it whenever indexing
logic changes in a way that could produce different results for
unchanged files, and every existing entry is treated as stale in one
step, the same fail-safe way a missing/corrupt cache already is.
"""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from repo_index.models import CallSite, FileIndex, ImportEdge, Symbol

_CACHE_FILENAME = "repo_index_cache.json"
_CACHE_VERSION = 1


def cache_path_for(root: Path) -> Path:
    return root / ".agent_state" / _CACHE_FILENAME


def load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict) or data.get("version") != _CACHE_VERSION:
        return {}  # missing/mismatched version -- treat as empty, force a full rebuild
    return data.get("entries", {})


def save_cache(path: Path, entries: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": _CACHE_VERSION, "entries": entries}), encoding="utf-8"
    )


def is_fresh(entry: dict, mtime: float, size: int) -> bool:
    return entry.get("mtime") == mtime and entry.get("size") == size


def file_index_to_entry(file_index: FileIndex, mtime: float, size: int) -> dict:
    return {
        "mtime": mtime,
        "size": size,
        "file_index": {
            "path": file_index.path,
            "language": file_index.language,
            "symbols": [asdict(s) for s in file_index.symbols],
            "imports": [asdict(i) for i in file_index.imports],
            "calls": [asdict(c) for c in file_index.calls],
        },
    }


def entry_to_file_index(entry: dict) -> FileIndex:
    data = entry["file_index"]
    return FileIndex(
        path=data["path"],
        language=data["language"],
        symbols=[Symbol(**s) for s in data["symbols"]],
        imports=[ImportEdge(**i) for i in data["imports"]],
        calls=[CallSite(**c) for c in data["calls"]],
    )
