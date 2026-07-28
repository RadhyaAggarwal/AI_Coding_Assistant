"""Repository-overview tool: gives the model a structural summary
(languages present, file count, detected manifest/config files) instead
of it having to infer the project's technologies from scratch. Backed by
repo_index's scanner (structure only, no AST parsing — cheaper than
find_symbol). Read-only, no arguments, always covers the whole project.
"""
from pathlib import Path
from typing import Any

from repo_index.scanner import scan
from tools.base import Tool


class RepoOverviewTool(Tool):
    name = "repo_overview"
    description = (
        "Get a structural overview of the project: file count, "
        "languages used (by file extension), and detected manifest/"
        "config files (requirements.txt, package.json, etc.). Takes no "
        "arguments. It always covers the whole project."
    )
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return "Getting a repo overview..."

    def run(self) -> str:
        summary = scan(self._project_root)
        languages = sorted(summary.languages.items(), key=lambda kv: -kv[1])
        lines = [
            f"Root: {summary.root}",
            f"Files: {summary.file_count}",
            "Languages: " + (", ".join(f"{lang} ({count})" for lang, count in languages) or "none detected"),
            "Manifests: " + (", ".join(summary.manifests) or "none found"),
        ]
        return "\n".join(lines)
