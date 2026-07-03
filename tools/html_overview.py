"""Per-file HTML structural overview: external script/stylesheet
references and an element count. Distinct from tools/find_symbol.py
because these aren't "find X by name" lookups — they're the kind of
cross-file relationship info the build plan calls out separately ("how
do different parts connect?").

Elements with an id attribute ARE indexed as symbols (kind="element",
"#id" name, matching CSS convention) and are findable via find_symbol
instead of being duplicated here.
"""
from pathlib import Path
from typing import Any

from repo_index.html_index import html_references
from tools.base import Tool
from tools.path_safety import resolve_within_root


class HtmlOverviewTool(Tool):
    name = "html_overview"
    description = (
        "Get a structural overview of a specific HTML file: which "
        "external <script src=...> and <link rel=stylesheet href=...> "
        "files it references, how many inline <script> blocks it has, "
        "and its total element count. Elements with an id attribute are "
        "findable separately via find_symbol using '#id' syntax."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to an HTML file, relative to the project root.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def run(self, path: str) -> str:
        resolved = resolve_within_root(self._project_root, path)
        if not resolved.is_file():
            raise FileNotFoundError(f"No such file: {path}")

        refs = html_references(resolved)
        lines = [
            f"Elements: {refs.element_count}",
            "Stylesheets: " + (", ".join(refs.stylesheets) or "none"),
            "Scripts (external): " + (", ".join(refs.scripts) or "none"),
            f"Inline <script> blocks: {refs.inline_script_count}",
        ]
        return "\n".join(lines)
