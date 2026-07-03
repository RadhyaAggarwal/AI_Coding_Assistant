"""CSS symbol indexer using tree-sitter — the non-Python language parsing
path CLAUDE.md requires (ast is Python-only; see python_index.py).

Extracts class/id selectors and @keyframes names as symbols. Selectors
are a weaker fit for the function/class/method vocabulary than Python or
JS give us, but they're still named, addressable units worth looking up
(e.g. "where is .button styled").
"""
from pathlib import Path

import tree_sitter_css
from tree_sitter import Language, Node, Parser

from repo_index.models import FileIndex, Symbol

_LANGUAGE = Language(tree_sitter_css.language())


def _find_child(node: Node, type_name: str) -> Node | None:
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def index_css_file(path: Path, relative_path: str) -> FileIndex:
    source = path.read_bytes()
    parser = Parser(_LANGUAGE)
    tree = parser.parse(source)

    symbols: list[Symbol] = []

    def walk(node: Node) -> None:
        if node.type in ("class_selector", "id_selector"):
            symbols.append(
                Symbol(
                    name=node.text.decode("utf-8"),
                    kind="selector",
                    file=relative_path,
                    line=node.start_point[0] + 1,
                )
            )
        elif node.type == "keyframes_statement":
            name_node = _find_child(node, "keyframes_name")
            if name_node is not None:
                symbols.append(
                    Symbol(
                        name=name_node.text.decode("utf-8"),
                        kind="keyframes",
                        file=relative_path,
                        line=node.start_point[0] + 1,
                    )
                )
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return FileIndex(path=relative_path, language="css", symbols=symbols)
