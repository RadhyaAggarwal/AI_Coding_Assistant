"""Lets the agent look up recent past requests and their outcomes, for a
request that references earlier work (e.g. "continue the pricing feature
from before") without the human needing to re-explain it. Backed by
agent_controller/session_log.py's lightweight, bounded log -- NOT a full
transcript, so this is a pointer to re-orient from, not a source of
truth about current file/code state. The description below tells the
model explicitly to re-verify anything it finds here against the real
project, the same fabrication-avoidance stance every other tool in this
project already takes toward partial/summary information.
"""
import time
from pathlib import Path
from typing import Any

from agent_controller.session_log import read_sessions
from tools.base import Tool

_DEFAULT_LIMIT = 10


class RecentActivityTool(Tool):
    name = "recent_activity"
    description = (
        "See a short log of recent past requests made to this agent and "
        "their outcomes, most recent last. Useful when the current "
        "request references earlier work without already knowing what "
        "was done. This is a bounded summary, not full detail -- "
        "re-verify anything it suggests against the actual current "
        "project state (read_file, find_symbol, or a git log via "
        "run_command) before relying on it; it reflects what was asked "
        "and a short outcome at the time, not necessarily what the code "
        "looks like now."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": f"How many recent sessions to return (default {_DEFAULT_LIMIT}).",
            },
        },
        "required": [],
    }

    def __init__(self, state_dir: str | Path):
        self._state_dir = Path(state_dir)

    def progress_message(self, arguments: dict[str, Any]) -> str:
        return "Checking recent activity..."

    def run(self, limit: int = _DEFAULT_LIMIT) -> str:
        entries = read_sessions(self._state_dir, limit=limit)
        if not entries:
            return "No recorded past sessions."
        lines = []
        for entry in entries:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry["timestamp"]))
            status = "done" if entry["completed"] else "incomplete"
            lines.append(
                f"{when} [{status}] asked: {entry['request']}\n  outcome: {entry['outcome']}"
            )
        return "\n".join(lines)
