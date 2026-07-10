"""File-reading tool. Path access is validated against the project root
before any read happens, so the agent can never escape the project
directory via '..' segments or an absolute path.
"""
from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.path_safety import PathOutsideProjectError, resolve_within_root

__all__ = ["PathOutsideProjectError", "ReadFileTool"]


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the full text contents of a file inside the project. "
        "The path must be relative to the project root."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path, relative to the project root.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return f"Reading {arguments['path']}..."

    def run(self, path: str) -> str:
        resolved = resolve_within_root(self._project_root, path)
        if not resolved.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        return resolved.read_text(encoding="utf-8")
