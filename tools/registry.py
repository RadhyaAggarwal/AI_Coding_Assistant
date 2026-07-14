"""Central registry mapping tool names to Tool instances and their
JSON schemas. This is the only place agent_controller/ looks up tools by
name — it never imports a tool class directly.
"""
from typing import Any, Callable

from state.snapshot import SnapshotManager
from tools.base import Tool
from tools.confirmation import ConfirmFn, ToolCallDeniedError, prompt_confirm
from tools.validation import validate_arguments

# Injectable the same way ConfirmFn is, so tests can capture progress
# messages instead of asserting on stdout.
ReportFn = Callable[[str], None]


class ToolRegistry:
    def __init__(
        self,
        confirm: ConfirmFn = prompt_confirm,
        snapshots: SnapshotManager | None = None,
        report: ReportFn = print,
    ):
        self._tools: dict[str, Tool] = {}
        self._confirm = confirm
        self._snapshots = snapshots
        self._report = report

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

    def dedup_exempt_names(self) -> set[str]:
        """Names of tools agent_controller/loop.py's duplicate-call guard
        should never block (see Tool.dedup_exempt) -- tools where human
        confirmation already gates every single invocation, making a
        harness-level repeat-block redundant at best.
        """
        return {tool.name for tool in self._tools.values() if tool.dedup_exempt}

    def always_mutates_names(self) -> set[str]:
        """Names of tools whose success guarantees a file was actually
        written (see Tool.always_mutates) -- used by
        agent_controller/loop.py to decide whether a success should
        invalidate every other cached tool result.
        """
        return {tool.name for tool in self._tools.values() if tool.always_mutates}

    def report(self, message: str) -> None:
        """Expose the same human-facing reporter used for tool progress
        messages (see Tool.progress_message()), so agent_controller/loop.py
        can surface loop-level events -- e.g. a duplicate call being
        rejected -- through the same channel instead of adding a second,
        separately-injected one.
        """
        self._report(message)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'")
        tool = self._tools[name]
        validate_arguments(tool.parameters, arguments)
        self._report(tool.progress_message(arguments))

        if tool.requires_confirmation:
            description = tool.confirmation_message(arguments)
            approval = self._confirm(description)
            # approval is bool | str (see tools/confirmation.py) -- must
            # check "is not True" explicitly, not "not approval", since a
            # non-empty feedback string is truthy and would otherwise be
            # read as approval.
            if approval is not True:
                reason = f": {approval}" if isinstance(approval, str) else ""
                raise ToolCallDeniedError(f"User declined to run '{name}' with {arguments}{reason}")

        if self._snapshots is not None:
            target = tool.target_path(arguments)
            if target is not None:
                self._snapshots.snapshot_before_edit(target)

        result = tool.run(**arguments)
        if tool.show_result:
            self._report(result)
        return result
