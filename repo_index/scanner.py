"""Repository scanner: walks the project, detects languages by file
extension, and identifies key manifest/config files (requirements.txt,
package.json, ...) as a lightweight signal of what technologies are
used. Not AST-aware — that's python_index.py / js_index.py's job.

Exposes the file-walking logic (iter_project_files) so indexer.py can
reuse the same skip rules instead of re-implementing them.
"""
from pathlib import Path
from typing import Iterator

from repo_index.models import RepoSummary

SKIP_DIR_NAMES = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".pytest_cache", ".agent_state",
}

EXTENSION_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}

_MANIFEST_NAMES = {
    "requirements.txt", "pyproject.toml", "setup.py", "Pipfile",
    "package.json", "Cargo.toml", "go.mod", "pom.xml", "build.gradle",
}


def iter_project_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        yield path


def scan(root: Path) -> RepoSummary:
    file_count = 0
    languages: dict[str, int] = {}
    manifests: list[str] = []

    for path in iter_project_files(root):
        file_count += 1
        language = EXTENSION_LANGUAGES.get(path.suffix.lower())
        if language:
            languages[language] = languages.get(language, 0) + 1
        if path.name in _MANIFEST_NAMES:
            manifests.append(str(path.relative_to(root)))

    return RepoSummary(
        root=str(root),
        file_count=file_count,
        languages=languages,
        manifests=sorted(manifests),
    )
