"""Symbol-lookup tool: finds where a function, method, class, CSS
selector, or HTML element id is defined by name, using repo_index's
AST-based index (ast for Python, tree-sitter for JavaScript/CSS/HTML)
rather than a plain text search — this only matches real definitions,
not every text occurrence of the name the way search_code would.
"""
from pathlib import Path
from typing import Any

from repo_index.indexer import RepoIndex
from tools.base import Tool
from tools.source_snippet import snippet_at


class FindSymbolTool(Tool):
    name = "find_symbol"
    description = (
        "Find where a function, method, class, CSS selector (e.g. "
        "'.button' or '#header'), or HTML element with an id (e.g. "
        "'#header') is defined in the project (case-insensitive exact "
        "match on the symbol name, not a substring search). Returns "
        "each match's file, line number, kind, and a short source "
        "snippet (a few lines starting at its definition). Only "
        "understands Python, JavaScript, CSS, and HTML files currently."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact symbol name to look up (e.g. a function or class name).",
            },
        },
        "required": ["name"],
    }

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return f"Looking up symbol '{arguments['name']}'..."

    def run(self, name: str) -> str:
        index = RepoIndex(self._project_root)
        matches = index.find_symbol(name)
        if not matches:
            return f"No symbol named '{name}' found."
        return "\n".join(
            f"{m.kind} {m.name} — {m.file}:{m.line}\n{snippet_at(self._project_root, m.file, m.line)}"
            for m in matches
        )
