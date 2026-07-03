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
class ImportEdge:
    importer_file: str  # file containing the import statement
    imported: str  # module/path being imported, as written in source
    line: int  # 1-indexed


@dataclass
class CallSite:
    caller_file: str  # file containing the call
    callee_name: str  # surface name of what's called — NOT resolved to a
    # specific class/object, so "obj.foo()" and "other.foo()" are
    # indistinguishable here (see tools/find_callers.py)
    line: int  # 1-indexed


@dataclass
class FileIndex:
    path: str  # relative to project root
    language: str  # "python" | "javascript" | ...
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[ImportEdge] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)


@dataclass
class RepoSummary:
    root: str
    file_count: int
    languages: dict[str, int]  # language -> file count
    manifests: list[str]  # detected manifest files, relative paths
