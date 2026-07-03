"""HTML indexer using tree-sitter — the non-Python language parsing path
CLAUDE.md requires (ast is Python-only; see python_index.py).

HTML elements don't map onto the function/class/selector "symbol" model
well in general, but elements with an id attribute do: an id is a named,
addressable unit, and the natural cross-reference point for CSS's #id
selectors — so those are indexed as symbols with kind="element", using
the "#id" convention to match css_index.py. Structural information that
ISN'T symbol-like — external script/stylesheet references — is handled
separately by html_references() below, used by tools/html_overview.py
rather than tools/find_symbol.py.
"""
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_html
from tree_sitter import Language, Node, Parser

from repo_index.models import FileIndex, Symbol

_LANGUAGE = Language(tree_sitter_html.language())


def _start_tag(node: Node) -> Node | None:
    return next((c for c in node.children if c.type == "start_tag"), None)


def _tag_name(start_tag: Node) -> str | None:
    name_node = next((c for c in start_tag.children if c.type == "tag_name"), None)
    return name_node.text.decode("utf-8") if name_node is not None else None


def _attribute_value(start_tag: Node, attr_name: str) -> str | None:
    for attribute in start_tag.children:
        if attribute.type != "attribute":
            continue
        name_node = next((c for c in attribute.children if c.type == "attribute_name"), None)
        if name_node is None or name_node.text.decode("utf-8") != attr_name:
            continue
        value_node = next((c for c in attribute.children if c.type == "quoted_attribute_value"), None)
        if value_node is None:
            return None
        inner = next((c for c in value_node.children if c.type == "attribute_value"), None)
        return inner.text.decode("utf-8") if inner is not None else None
    return None


def index_html_file(path: Path, relative_path: str) -> FileIndex:
    source = path.read_bytes()
    parser = Parser(_LANGUAGE)
    tree = parser.parse(source)

    symbols: list[Symbol] = []

    def walk(node: Node) -> None:
        if node.type in ("element", "script_element", "style_element"):
            start_tag = _start_tag(node)
            if start_tag is not None:
                element_id = _attribute_value(start_tag, "id")
                if element_id:
                    symbols.append(
                        Symbol(
                            name=f"#{element_id}",
                            kind="element",
                            file=relative_path,
                            line=node.start_point[0] + 1,
                        )
                    )
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return FileIndex(path=relative_path, language="html", symbols=symbols)


@dataclass
class HtmlReferences:
    scripts: list[str]  # external <script src=...> values
    stylesheets: list[str]  # external <link rel=stylesheet href=...> values
    inline_script_count: int
    element_count: int


def html_references(path: Path) -> HtmlReferences:
    source = path.read_bytes()
    parser = Parser(_LANGUAGE)
    tree = parser.parse(source)

    scripts: list[str] = []
    stylesheets: list[str] = []
    inline_script_count = 0
    element_count = 0

    def walk(node: Node) -> None:
        nonlocal inline_script_count, element_count
        if node.type in ("element", "script_element", "style_element"):
            element_count += 1

        if node.type == "script_element":
            start_tag = _start_tag(node)
            src = _attribute_value(start_tag, "src") if start_tag is not None else None
            if src:
                scripts.append(src)
            else:
                inline_script_count += 1
        elif node.type == "element":
            start_tag = _start_tag(node)
            if start_tag is not None and _tag_name(start_tag) == "link":
                rel = _attribute_value(start_tag, "rel")
                href = _attribute_value(start_tag, "href")
                if rel == "stylesheet" and href:
                    stylesheets.append(href)

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return HtmlReferences(
        scripts=scripts,
        stylesheets=stylesheets,
        inline_script_count=inline_script_count,
        element_count=element_count,
    )
