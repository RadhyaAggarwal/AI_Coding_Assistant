"""Orchestrates a full repository index: scans the project, dispatches
each file to the right language-specific indexer, and aggregates results
into one flat symbol table for lookup by tools/find_symbol.py and
tools/repo_overview.py.

No caching yet — each RepoIndex() call re-walks and re-parses the whole
project. Fine at this project's current size; worth revisiting if this
becomes slow on a larger repo.
"""
from pathlib import Path

from repo_index.css_index import index_css_file
from repo_index.html_index import index_html_file
from repo_index.js_index import index_javascript_file
from repo_index.models import FileIndex, RepoSummary, Symbol
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
        for path in iter_project_files(self._root):
            language = EXTENSION_LANGUAGES.get(path.suffix.lower())
            indexer = _INDEXERS.get(language)
            if indexer is None:
                continue
            relative_path = str(path.relative_to(self._root))
            try:
                self.files[relative_path] = indexer(path, relative_path)
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

    def find_symbol(self, name: str) -> list[Symbol]:
        needle = name.lower()
        matches: list[Symbol] = []
        for file_index in self.files.values():
            for symbol in file_index.symbols:
                if symbol.name.lower() == needle:
                    matches.append(symbol)
        return matches
