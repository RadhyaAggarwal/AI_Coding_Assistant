"""Base class every tool must implement.

The agent never touches the filesystem or shell directly — it always goes
through a Tool registered here, so all access can be validated in one
place.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the tool's arguments

    # Tools with real side effects beyond reading (e.g. running a shell
    # command) should set this True; ToolRegistry.execute() then requires
    # human confirmation before calling run(). Defaults False since most
    # tools are read-only and path-contained.
    requires_confirmation: bool = False

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return its result as a string."""
        raise NotImplementedError

    def target_path(self, arguments: dict[str, Any]) -> Path | None:
        """Absolute path this call is about to modify, if any.

        Tools that write files override this so ToolRegistry can snapshot
        the target first (see state/snapshot.py) before run() executes.
        Defaults to None for read-only tools, which need no snapshot.
        """
        return None

    def confirmation_message(self, arguments: dict[str, Any]) -> str:
        """Human-facing description shown before a requires_confirmation
        tool runs (see ToolRegistry.execute()). Defaults to a generic
        description of the raw call; override when a tool's consequences
        aren't obvious from its arguments alone (see tools/create_file.py
        for an example: warns explicitly when a call would overwrite an
        existing file's entire content, not just echo the raw arguments).
        """
        return f"Agent wants to run '{self.name}' with arguments {arguments}"

    def progress_message(self, arguments: dict[str, Any]) -> str:
        """Human-facing one-liner printed right before every call to this
        tool, confirmation-gated or not (see ToolRegistry.execute()).

        Unlike confirmation_message(), this prints for read-only tools
        too — those currently execute silently, so a multi-step run's
        read_file/find_symbol/etc. calls are invisible in the transcript
        even though they consume step budget. Observed live: a request
        that made a correct fix (visible via its two confirmation-gated
        edit_file/run_command calls) still ran out of step budget with no
        way to tell what the other steps were doing. Override with a
        natural-language description of the specific call (e.g. "Reading
        {path}...") when the generic fallback below isn't clear enough.
        """
        return f"Running {self.name}..."

    def schema(self) -> dict[str, Any]:
        """JSON schema describing this tool, as sent to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
