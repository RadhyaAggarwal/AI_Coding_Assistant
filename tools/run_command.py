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
import sys
from pathlib import Path
from typing import Any

from tools.base import Tool


def _kill_process_tree(process: "subprocess.Popen[str]") -> None:
    """Kill process and every descendant it spawned, not just the one
    handle subprocess itself tracks.

    Live-observed gap: subprocess.run(..., timeout=N) on Windows only
    guarantees killing the immediate child -- under shell=True that's
    cmd.exe. A command that spawns its own child (Django's runserver
    always does, via its auto-reloader) leaves that grandchild running
    as an orphan after the configured timeout supposedly fires, holding
    the port and sometimes the output pipes open -- twice observed live
    to leave the whole harness process itself stuck well past the
    configured timeout, requiring a human to manually find and kill the
    stray processes. taskkill's /T flag kills a process and its full
    descendant tree by walking Windows' own parent-child PID records, no
    extra dependency needed. On POSIX, the equivalent is killing the
    process group instead of just the one PID -- see start_new_session
    in run() below, which is what makes that safe (without it, the
    child's process group would be the same as this harness's own).
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
        )
    else:
        import os
        import signal

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    # The tree is dead now, so this won't hang the way the original bug
    # did -- reap the process so it doesn't linger as a zombie.
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        pass

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
        popen_kwargs: dict[str, Any] = {}
        if sys.platform != "win32":
            # Starts the child in its own process group/session, so
            # _kill_process_tree's os.killpg() can safely target just this
            # command's descendants on timeout -- without this, the
            # child's process group would be the same as this harness's
            # own, and killpg would kill us too.
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(
            command,
            shell=True,
            cwd=self._project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **popen_kwargs,
        )
        try:
            stdout, stderr = process.communicate(timeout=self._timeout_seconds)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            return f"Command timed out after {self._timeout_seconds}s: {command}"

        # Deliberately not truncated here -- agent_controller/context_budget.py's
        # cap_observation() is the one place that policy should live (see its
        # docstring for why a second, separately-drifting copy caused a real,
        # live-observed bug: this tool's own head-only cap discarded a
        # traceback's actual exception line before cap_observation's
        # head+tail fix ever got a chance to run).
        output = (stdout + stderr).strip()
        if output:
            return f"Exit code: {process.returncode}\n{output}"
        return f"Exit code: {process.returncode} (no output)"
