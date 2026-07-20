"""Shell command execution tool.

This is the one tool with real, uncontained side effects, so
requires_confirmation is True — ToolRegistry.execute() gates it behind
human approval before it runs (see tools/confirmation.py). There is no
sandboxing beyond that: the working directory is pinned to the project
root, output is truncated, and a timeout is enforced, but a *confirmed*
command can still do anything a human running it in this directory
could do.

Confirmation is this tool's real safety boundary, deliberately -- it
doesn't refuse commands outright, since a human might genuinely want to
run any of them. But that boundary only works if the human actually
notices what they're approving. Observed live: a bigger, more capable
model, given only "fix this bug and run the tests," went on unprompted
to run `git add` and then `git push origin main` -- entirely outside
the scope of what was asked. The push was correctly declined, but it
was buried in a routine-looking confirmation prompt alongside several
harmless ones, easy to approve on autopilot. confirmation_message()
below makes a small, known set of especially consequential command
shapes (pushing to a remote, discarding history/changes irreversibly,
recursive force-deletes) impossible to miss, without blocking anything
a human still chooses to approve.
"""
import re
import subprocess
from pathlib import Path
from typing import Any

from tools.base import Tool

_DANGEROUS_COMMAND_WARNINGS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"\bgit\s+push\b", re.IGNORECASE),
        "this pushes to a remote repository -- changes become visible to "
        "others and are hard to fully undo",
    ),
    (
        re.compile(r"\bgit\s+commit\b", re.IGNORECASE),
        "this creates a permanent commit in git history",
    ),
    (
        re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
        "this discards uncommitted local changes irreversibly",
    ),
    (
        re.compile(r"\brm\s+(-\S*r\S*f\S*|-\S*f\S*r\S*|-[rf]\s+-[rf])\b", re.IGNORECASE),
        "this recursively force-deletes files/directories -- not "
        "recoverable via this project's snapshot system",
    ),
]


def _looks_like_powershell_recurse_force_delete(command: str) -> bool:
    lowered = command.lower()
    return "remove-item" in lowered and "-recurse" in lowered and "-force" in lowered


def _dangerous_command_warnings(command: str) -> list[str]:
    warnings = [msg for pattern, msg in _DANGEROUS_COMMAND_WARNINGS if pattern.search(command)]
    if _looks_like_powershell_recurse_force_delete(command):
        warnings.append(
            "this recursively force-deletes files/directories -- not "
            "recoverable via this project's snapshot system"
        )
    return warnings


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Run a shell command (e.g. to run tests) inside the project "
        "directory. Requires human confirmation before it runs. Output "
        "is truncated and the command is killed if it exceeds the "
        "timeout."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run.",
            },
        },
        "required": ["command"],
    }
    requires_confirmation = True
    # See Tool.show_result -- a human should see real command output
    # (e.g. actual pytest results), not only the model's own narration
    # of what it showed.
    show_result = True
    # See Tool.dedup_exempt -- confirmation already gates every call to
    # this tool, so a repeat (e.g. re-running the tests to confirm a fix)
    # should reach that prompt, not get silently refused beforehand.
    dedup_exempt = True

    def __init__(self, project_root: str | Path, timeout_seconds: float = 60):
        self._project_root = Path(project_root).resolve()
        self._timeout_seconds = timeout_seconds

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return f"Running command: {arguments.get('command', '')}"

    def confirmation_message(self, arguments: dict[str, Any]) -> str:
        command = arguments.get("command", "")
        warnings = _dangerous_command_warnings(command)
        if not warnings:
            return f"Agent wants to run: {command}"
        warning_lines = "\n".join(f"  WARNING: {w}" for w in warnings)
        return f"Agent wants to run: {command}\n{warning_lines}"

    def run(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self._project_root,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self._timeout_seconds}s: {command}"

        # Deliberately not truncated here -- agent_controller/context_budget.py's
        # cap_observation() is the one place that policy should live (see its
        # docstring for why a second, separately-drifting copy caused a real,
        # live-observed bug: this tool's own head-only cap discarded a
        # traceback's actual exception line before cap_observation's
        # head+tail fix ever got a chance to run).
        output = (result.stdout + result.stderr).strip()
        if output:
            return f"Exit code: {result.returncode}\n{output}"
        return f"Exit code: {result.returncode} (no output)"
