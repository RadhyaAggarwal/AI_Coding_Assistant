"""Central registry mapping tool names to Tool instances and their
JSON schemas. This is the only place agent_controller/ looks up tools by
name — it never imports a tool class directly.
"""
from typing import Any

from tools.base import Tool
from tools.confirmation import ConfirmFn, ToolCallDeniedError, prompt_confirm
from tools.validation import validate_arguments


class ToolRegistry:
    def __init__(self, confirm: ConfirmFn = prompt_confirm):
        self._tools: dict[str, Tool] = {}
        self._confirm = confirm

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'")
        tool = self._tools[name]
        validate_arguments(tool.parameters, arguments)

        if tool.requires_confirmation:
            description = f"Agent wants to run '{name}' with arguments {arguments}"
            if not self._confirm(description):
                raise ToolCallDeniedError(f"User declined to run '{name}' with {arguments}")

        return tool.run(**arguments)
