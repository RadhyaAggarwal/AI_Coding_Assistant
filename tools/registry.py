"""Central registry mapping tool names to Tool instances and their
JSON schemas. This is the only place agent_controller/ looks up tools by
name — it never imports a tool class directly.
"""
from typing import Any

from state.snapshot import SnapshotManager
from tools.base import Tool
from tools.confirmation import ConfirmFn, ToolCallDeniedError, prompt_confirm
from tools.validation import validate_arguments


class ToolRegistry:
    def __init__(
        self,
        confirm: ConfirmFn = prompt_confirm,
        snapshots: SnapshotManager | None = None,
    ):
        self._tools: dict[str, Tool] = {}
        self._confirm = confirm
        self._snapshots = snapshots

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def confirmation_required_names(self) -> set[str]:
        """Names of the tools requiring human confirmation to run (see
        Tool.requires_confirmation) — the side-effecting minority whose
        silent exclusion from a narrowed/routed tool list would break a
        task, unlike the read-only majority. Used by
        agent_controller.tool_router to protect them from keyword-overlap
        scoring regardless of a request's wording.
        """
        return {tool.name for tool in self._tools.values() if tool.requires_confirmation}

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'")
        tool = self._tools[name]
        validate_arguments(tool.parameters, arguments)

        if tool.requires_confirmation:
            description = f"Agent wants to run '{name}' with arguments {arguments}"
            if not self._confirm(description):
                raise ToolCallDeniedError(f"User declined to run '{name}' with {arguments}")

        if self._snapshots is not None:
            target = tool.target_path(arguments)
            if target is not None:
                self._snapshots.snapshot_before_edit(target)

        return tool.run(**arguments)
