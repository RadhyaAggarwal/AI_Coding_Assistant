"""Orchestrates a full repository index: scans the project, dispatches
each file to the right language-specific indexer, and aggregates results
into one flat symbol table for lookup by tools/find_symbol.py,
tools/find_importers.py, tools/find_callers.py, and tools/repo_overview.py.

Per-file parse results are cached to disk (repo_index/cache.py), keyed
by each file's mtime and size, so an unchanged file is never re-parsed —
only re-walked (a cheap stat call) — on the next RepoIndex() build,
whether that's later in the same request or a separate CLI invocation.
"""
from pathlib import Path

from repo_index.cache import (
    cache_path_for,
    entry_to_file_index,
    file_index_to_entry,
    is_fresh,
    load_cache,
    save_cache,
)
from repo_index.css_index import index_css_file
from repo_index.html_index import index_html_file
from repo_index.js_index import index_javascript_file
from repo_index.models import CallSite, FileIndex, ImportEdge, RepoSummary, Symbol
from repo_index.python_index import index_python_file
from repo_index.scanner import EXTENSION_LANGUAGES, iter_project_files, scan

_INDEXERS = {
    "python": index_python_file,
    "javascript": index_javascript_file,
    "css": index_css_file,
    "html": index_html_file,
}


class RepoIndex:
    def __init__(self, root: str | Path):
        self._root = Path(root).resolve()
        self.summary: RepoSummary = scan(self._root)
        self.files: dict[str, FileIndex] = {}
        self._build()

    def _build(self) -> None:
        cache_file = cache_path_for(self._root)
        cached_entries = load_cache(cache_file)
        fresh_entries: dict[str, dict] = {}

        for path in iter_project_files(self._root):
            language = EXTENSION_LANGUAGES.get(path.suffix.lower())
            indexer = _INDEXERS.get(language)
            if indexer is None:
                continue
            relative_path = str(path.relative_to(self._root))
            try:
                stat = path.stat()
            except OSError:
                continue

            cached = cached_entries.get(relative_path)
            if cached is not None and is_fresh(cached, stat.st_mtime, stat.st_size):
                try:
                    self.files[relative_path] = entry_to_file_index(cached)
                    fresh_entries[relative_path] = cached
                    continue
                except (KeyError, TypeError):
                    pass  # corrupt or incompatible cache entry -- fall through to re-parse

            try:
                file_index = indexer(path, relative_path)
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            self.files[relative_path] = file_index
            fresh_entries[relative_path] = file_index_to_entry(file_index, stat.st_mtime, stat.st_size)

        save_cache(cache_file, fresh_entries)

    def find_symbol(self, name: str) -> list[Symbol]:
        needle = name.lower()
        matches: list[Symbol] = []
        for file_index in self.files.values():
            for symbol in file_index.symbols:
                if symbol.name.lower() == needle:
                    matches.append(symbol)
        return matches

    def find_importers(self, module_name: str) -> list[ImportEdge]:
        needle = module_name.lower()
        matches: list[ImportEdge] = []
        for file_index in self.files.values():
            for edge in file_index.imports:
                if edge.imported.lower() == needle:
                    matches.append(edge)
        return matches

    def find_callers(self, function_name: str) -> list[CallSite]:
        needle = function_name.lower()
        matches: list[CallSite] = []
        for file_index in self.files.values():
            for call in file_index.calls:
                if call.callee_name.lower() == needle:
                    matches.append(call)
        return matches
