"""Model-agnostic interface that every LLM backend must implement.

No model-specific (Qwen, DeepSeek, etc.) logic belongs in this file or in
any code outside model_interface/ — callers only ever see this interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """Normalized result of a generate() call, regardless of backend."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None


class ModelInterface(ABC):
    """Abstract base class for talking to a local LLM.

    Implementations wrap a specific inference backend (Ollama, llama.cpp
    server, vLLM, ...) behind this same interface so that
    agent_controller/ and tools/ never depend on backend-specific details.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """Send a prompt to the model and return its response.

        Args:
            prompt: The full prompt text to send to the model.
            tools: Optional list of JSON-schema tool definitions the model
                may choose to call, as produced by the tool registry.

        Returns:
            A ModelResponse containing the model's text and any tool
            calls it requested.
        """
        raise NotImplementedError
