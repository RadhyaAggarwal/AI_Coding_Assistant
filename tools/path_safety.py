"""Shared path-containment check used by every tool that touches the
project directory (read_file, list_directory, search_code, ...), so the
agent can never escape the project root via '..' segments or an absolute
path. One implementation, reused, instead of each tool reinventing it.
"""
from pathlib import Path


class PathOutsideProjectError(Exception):
    """Raised when a requested path resolves outside the project root."""


def resolve_within_root(project_root: Path, path: str) -> Path:
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise PathOutsideProjectError(
            f"Refusing to access '{path}': resolves outside the project root."
        )
    return resolved
