"""Meaning-based code search: finds relevant code for a vague, natural-
language question, using repo_index/semantic_index.py's embeddings index
rather than an exact name or literal text match.

Complements, not replaces, find_symbol/search_code -- those stay faster
and more precise whenever an exact symbol name or literal text is
already known; this exists specifically for "I don't know what this is
called, only what it does" queries, a real, repeatedly-observed gap
those exact-match tools can't help with.
"""
from pathlib import Path
from typing import Any, Callable

from model_interface.base import ModelInterface
from repo_index.semantic_index import SemanticIndex
from tools.base import Tool


class SemanticSearchTool(Tool):
    name = "semantic_search"
    description = (
        "Find code relevant to a vague, natural-language question about "
        "what something does or how it works, when you don't know its "
        "exact name (e.g. 'how do we make sure an uploaded file is not "
        "too large', 'where do we handle a failed payment retry'). Ranks "
        "real functions/classes by meaning, not literal text match. If "
        "you already know the exact symbol name or a literal string to "
        "search for, use find_symbol or search_code instead -- they're "
        "faster and more precise for that. Requires an embedding model "
        "to be configured; unavailable otherwise."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A natural-language description of what you're looking for.",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        project_root: str | Path,
        model: ModelInterface,
        report: Callable[[str], None] = print,
    ):
        self._project_root = Path(project_root).resolve()
        self._model = model
        # Optional-with-a-default the same way ToolRegistry's own report
        # callback is -- see Tool.progress_message()'s docstring on why
        # per-call messages matter, and repo_index/semantic_index.py's
        # SemanticIndex.on_progress for why this one specifically needs
        # more than the usual single message: building/updating the
        # index can take real, human-noticeable time (embedding every
        # new/changed chunk), unlike every other tool's near-instant
        # run(). Defaulting to print (not a no-op) means this is visible
        # in the real CLI without any extra wiring; tests can still
        # inject a capturing callback the same way ToolRegistry's tests
        # already do.
        self._report = report

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return f"Searching semantically for '{arguments['query']}'..."

    def run(self, query: str) -> str:
        index = SemanticIndex(self._project_root, self._model, on_progress=self._report)
        results = index.search(query)
        if not results:
            return "No indexed code found to search (the project may have no supported files yet)."
        return "\n\n".join(
            f"{chunk.name} — {chunk.file}:{chunk.start_line}-{chunk.end_line} "
            f"(similarity {score:.2f})\n"
            + "\n".join(f"    {line}" for line in chunk.text.splitlines())
            for chunk, score in results
        )
