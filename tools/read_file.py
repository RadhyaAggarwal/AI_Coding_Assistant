"""File-reading tool. Path access is validated against the project root
before any read happens, so the agent can never escape the project
directory via '..' segments or an absolute path.
"""
from pathlib import Path
from typing import Any

from tools.base import Tool


class PathOutsideProjectError(Exception):
    """Raised when a requested path resolves outside the project root."""


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

    def run(self, path: str) -> str:
        resolved = (self._project_root / path).resolve()
        try:
            resolved.relative_to(self._project_root)
        except ValueError:
            raise PathOutsideProjectError(
                f"Refusing to read '{path}': resolves outside the project root."
            )
        if not resolved.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        return resolved.read_text(encoding="utf-8")
