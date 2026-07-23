"""Directory-listing tool. Path access is validated against the project
root the same way read_file's is, so the agent can only browse inside
the project.
"""
from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.path_safety import resolve_within_root
from tools.path_suggestions import suggest_similar_directories


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = (
        "List the files and subdirectories directly inside a directory "
        "in the project. The path must be relative to the project root "
        "('.' for the project root itself)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path, relative to the project root.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return f"Listing {arguments['path']}..."

    def run(self, path: str) -> str:
        resolved = resolve_within_root(self._project_root, path)
        if not resolved.is_dir():
            message = f"No such directory: {path}"
            suggestions = suggest_similar_directories(self._project_root, path)
            if suggestions:
                message += ". Did you mean one of these real directories? " + ", ".join(suggestions)
            raise NotADirectoryError(message)

        entries = sorted(resolved.iterdir(), key=lambda entry: entry.name)
        if not entries:
            return "(empty directory)"
        return "\n".join(
            f"{'d' if entry.is_dir() else 'f'}  {entry.name}" for entry in entries
        )
