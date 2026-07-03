"""Python symbol indexer using the standard library ast module.

Per CLAUDE.md: ast is for Python-specific analysis; non-Python languages
go through tree-sitter instead (see js_index.py). Captures three kinds
of information: definitions (Symbol), import statements (ImportEdge),
and call sites (CallSite). Call sites are matched on surface name only —
"obj.foo()" is recorded as a call to "foo" regardless of what obj is;
resolving the actual target would need real type inference, out of scope.
"""
import ast
from pathlib import Path

from repo_index.models import CallSite, FileIndex, ImportEdge, Symbol


def index_python_file(path: Path, relative_path: str) -> FileIndex:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path)

    symbols: list[Symbol] = []
    imports: list[ImportEdge] = []
    calls: list[CallSite] = []

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

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                imports.append(
                    ImportEdge(importer_file=relative_path, imported=alias.name, line=node.lineno)
                )
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = ("." * node.level) + (node.module or "")
            imports.append(
                ImportEdge(importer_file=relative_path, imported=module, line=node.lineno)
            )
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            callee = None
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            if callee:
                calls.append(CallSite(caller_file=relative_path, callee_name=callee, line=node.lineno))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return FileIndex(path=relative_path, language="python", symbols=symbols, imports=imports, calls=calls)
