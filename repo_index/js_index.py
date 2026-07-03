"""JavaScript symbol indexer using tree-sitter — the non-Python language
parsing path CLAUDE.md requires (ast is Python-only; see python_index.py).

Extracts top-level function/class/method declarations. Intentionally a
shallow first pass — no exports, no arrow-function-assigned-to-const
detection yet — not a full JS analyzer.
"""
from pathlib import Path

import tree_sitter_javascript
from tree_sitter import Language, Node, Parser

from repo_index.models import FileIndex, Symbol

_LANGUAGE = Language(tree_sitter_javascript.language())

_NODE_TYPE_KINDS = {
    "function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
}


def _find_identifier(node: Node) -> str | None:
    for child in node.children:
        if child.type in ("identifier", "property_identifier"):
            return child.text.decode("utf-8")
    return None


def index_javascript_file(path: Path, relative_path: str) -> FileIndex:
    source = path.read_bytes()
    parser = Parser(_LANGUAGE)
    tree = parser.parse(source)

    symbols: list[Symbol] = []

    def walk(node: Node) -> None:
        kind = _NODE_TYPE_KINDS.get(node.type)
        if kind:
            name = _find_identifier(node)
            if name:
                symbols.append(
                    Symbol(name=name, kind=kind, file=relative_path, line=node.start_point[0] + 1)
                )
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return FileIndex(path=relative_path, language="javascript", symbols=symbols)
