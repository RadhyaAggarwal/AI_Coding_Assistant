"""Base class every tool must implement.

The agent never touches the filesystem or shell directly — it always goes
through a Tool registered here, so all access can be validated in one
place.
"""
from abc import ABC, abstractmethod
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
