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
