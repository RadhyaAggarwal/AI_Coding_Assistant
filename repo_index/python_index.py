"""Python symbol indexer using the standard library ast module.

Per CLAUDE.md: ast is for Python-specific analysis; non-Python languages
go through tree-sitter instead (see js_index.py).
"""
import ast
from pathlib import Path

from repo_index.models import FileIndex, Symbol


def index_python_file(path: Path, relative_path: str) -> FileIndex:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path)

    symbols: list[Symbol] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self._class_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            kind = "method" if self._class_depth > 0 else "function"
            symbols.append(Symbol(name=node.name, kind=kind, file=relative_path, line=node.lineno))
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            symbols.append(Symbol(name=node.name, kind="class", file=relative_path, line=node.lineno))
            self._class_depth += 1
            self.generic_visit(node)
            self._class_depth -= 1

    _Visitor().visit(tree)
    return FileIndex(path=relative_path, language="python", symbols=symbols)
