"""Call-site lookup: where a function or method is actually called, not
just where its name appears as text (unlike search_code) and not where
it's defined (that's find_symbol). Matches on the call's surface name
only — "obj1.foo()" and "obj2.foo()" both match a search for "foo"; this
doesn't resolve which class/object a method call belongs to (real
resolution would need type inference, out of scope).
"""
from pathlib import Path
from typing import Any

from repo_index.indexer import RepoIndex
from tools.base import Tool
from tools.source_snippet import snippet_at


class FindCallersTool(Tool):
    name = "find_callers"
    description = (
        "Find where a function or method named 'name' is actually "
        "called (case-insensitive exact match on the call's surface "
        "name, not a substring search, and not resolved to a specific "
        "class/object — 'obj.foo()' matches a search for 'foo' "
        "regardless of what obj is). Returns each call site's file, "
        "line number, and a short source snippet. Only understands "
        "Python and JavaScript files."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Function or method name to find calls to.",
            },
        },
        "required": ["name"],
    }

    def __init__(self, project_root: str | Path):
        self._project_root = Path(project_root).resolve()

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return f"Finding callers of '{arguments['name']}'..."

    def run(self, name: str) -> str:
        index = RepoIndex(self._project_root)
        matches = index.find_callers(name)
        if not matches:
            return f"No calls to '{name}' found."
        return "\n".join(
            f"{m.caller_file}:{m.line}\n{snippet_at(self._project_root, m.caller_file, m.line)}"
            for m in matches
        )
