"""Import-relationship lookup: which files import a given module. Uses
repo_index's per-file import extraction (ast for Python; tree-sitter for
JavaScript, including CommonJS require(...) calls) — an exact match on
the module reference as written in the source (e.g. "os.path",
"./utils.js"), not a substring search.
"""
from pathlib import Path
from typing import Any

from repo_index.indexer import RepoIndex
from tools.base import Tool
from tools.source_snippet import snippet_at


class FindImportersTool(Tool):
    name = "find_importers"
    description = (
        "Find which files import a given module (case-insensitive exact "
        "match on the import reference as written in source, e.g. "
        "'os.path' or './utils.js', not a substring search). Covers "
        "Python import/from-import statements and JavaScript "
        "import/require() statements. Returns each match's file, line "
        "number, and a short source snippet."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "module_name": {
                "type": "string",
                "description": "The module/path to search for, exactly as it would appear in an import statement.",
            },
        },
        "required": ["module_name"],
    }

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return f"Finding importers of '{arguments['module_name']}'..."

    def run(self, module_name: str) -> str:
        index = RepoIndex(self._project_root)
        matches = index.find_importers(module_name)
        if not matches:
            return f"No file imports '{module_name}'."
        return "\n".join(
            f"{m.importer_file}:{m.line}\n{snippet_at(self._project_root, m.importer_file, m.line)}"
            for m in matches
        )
