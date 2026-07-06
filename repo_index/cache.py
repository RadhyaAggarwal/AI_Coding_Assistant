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
"""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from repo_index.models import CallSite, FileIndex, ImportEdge, Symbol

_CACHE_FILENAME = "repo_index_cache.json"


def cache_path_for(root: Path) -> Path:
    return root / ".agent_state" / _CACHE_FILENAME


def load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(path: Path, entries: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")


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
