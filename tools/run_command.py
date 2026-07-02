"""Shell command execution tool.

This is the one tool with real, uncontained side effects, so
requires_confirmation is True — ToolRegistry.execute() gates it behind
human approval before it runs (see tools/confirmation.py). There is no
sandboxing beyond that: the working directory is pinned to the project
root, output is truncated, and a timeout is enforced, but a *confirmed*
command can still do anything a human running it in this directory
could do.
"""
import subprocess
from pathlib import Path
from typing import Any

from tools.base import Tool

_MAX_OUTPUT_CHARS = 4000


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

    def __init__(self, project_root: str | Path, timeout_seconds: float = 60):
        self._project_root = Path(project_root).resolve()
        self._timeout_seconds = timeout_seconds

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

        output = (result.stdout + result.stderr).strip()
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"

        if output:
            return f"Exit code: {result.returncode}\n{output}"
        return f"Exit code: {result.returncode} (no output)"
