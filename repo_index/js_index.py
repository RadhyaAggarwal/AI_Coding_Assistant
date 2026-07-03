"""JavaScript symbol indexer using tree-sitter — the non-Python language
parsing path CLAUDE.md requires (ast is Python-only; see python_index.py).

Extracts top-level function/class/method declarations (Symbol), import
statements including CommonJS require(...) (ImportEdge), and call sites
(CallSite). Call sites are matched on surface name only — "obj.foo()" is
recorded as a call to "foo" regardless of what obj is; resolving the
actual target would need real type inference, out of scope. Intentionally
a shallow first pass overall — no exports, no arrow-function-assigned-
to-const detection yet — not a full JS analyzer.
"""
from pathlib import Path

import tree_sitter_javascript
from tree_sitter import Language, Node, Parser

from repo_index.models import CallSite, FileIndex, ImportEdge, Symbol

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


def _first_string_value(node: Node) -> str | None:
    if node.type == "string":
        fragment = next((c for c in node.children if c.type == "string_fragment"), None)
        return fragment.text.decode("utf-8") if fragment is not None else None
    for child in node.children:
        result = _first_string_value(child)
        if result is not None:
            return result
    return None


def _call_callee_name(node: Node) -> str | None:
    func_node = node.children[0] if node.children else None
    if func_node is None:
        return None
    if func_node.type == "identifier":
        return func_node.text.decode("utf-8")
    if func_node.type == "member_expression":
        prop = next((c for c in func_node.children if c.type == "property_identifier"), None)
        return prop.text.decode("utf-8") if prop is not None else None
    return None


def index_javascript_file(path: Path, relative_path: str) -> FileIndex:
    source = path.read_bytes()
    parser = Parser(_LANGUAGE)
    tree = parser.parse(source)

    symbols: list[Symbol] = []
    imports: list[ImportEdge] = []
    calls: list[CallSite] = []

    def walk(node: Node) -> None:
        kind = _NODE_TYPE_KINDS.get(node.type)
        if kind:
            name = _find_identifier(node)
            if name:
                symbols.append(
                    Symbol(name=name, kind=kind, file=relative_path, line=node.start_point[0] + 1)
                )

        if node.type == "import_statement":
            source_path = _first_string_value(node)
            if source_path:
                imports.append(
                    ImportEdge(importer_file=relative_path, imported=source_path, line=node.start_point[0] + 1)
                )

        if node.type == "call_expression":
            callee = _call_callee_name(node)
            if callee == "require":
                args_node = next((c for c in node.children if c.type == "arguments"), None)
                source_path = _first_string_value(args_node) if args_node is not None else None
                if source_path:
                    imports.append(
                        ImportEdge(importer_file=relative_path, imported=source_path, line=node.start_point[0] + 1)
                    )
            elif callee:
                calls.append(CallSite(caller_file=relative_path, callee_name=callee, line=node.start_point[0] + 1))

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return FileIndex(path=relative_path, language="javascript", symbols=symbols, imports=imports, calls=calls)
