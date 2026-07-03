"""Language-agnostic data model shared by every language-specific indexer
(python_index.py via ast, js_index.py via tree-sitter, ...), so the
orchestrator and the tools that wrap it never need to know which parser
produced a given result.
"""
from dataclasses import dataclass, field


@dataclass
class Symbol:
    name: str
    kind: str  # "function" | "method" | "class"
    file: str  # path relative to project root
    line: int  # 1-indexed


@dataclass
class FileIndex:
    path: str  # relative to project root
    language: str  # "python" | "javascript" | ...
    symbols: list[Symbol] = field(default_factory=list)


@dataclass
class RepoSummary:
    root: str
    file_count: int
    languages: dict[str, int]  # language -> file count
    manifests: list[str]  # detected manifest files, relative paths
