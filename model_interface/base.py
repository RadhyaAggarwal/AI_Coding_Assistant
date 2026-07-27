"""Model-agnostic interface that every LLM backend must implement.

No model-specific (Qwen, DeepSeek, etc.) logic belongs in this file or in
any code outside model_interface/ — callers only ever see this interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ModelUnavailableError(Exception):
    """Raised when a backend can't be reached or is unresponsive.

    Distinct from a normal generation failure: a stuck backend process can
    block a full-length request timeout on every call until a human
    intervenes, so callers should surface this promptly instead of letting
    each call hang for the full request timeout in turn.
    """


@dataclass
class Message:
    """One turn in a chat-style conversation.

    role is one of "system", "user", "assistant", "tool" — the roles used
    by essentially every chat-tuned LLM API, not specific to any backend.
    """

    role: str
    content: str


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
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """Send a chat-style conversation to the model and return its response.

        Args:
            messages: The full conversation so far, oldest first (typically
                a system message, then alternating user/assistant/tool
                turns).
            tools: Optional list of JSON-schema tool definitions the model
                may choose to call, as produced by the tool registry.

        Returns:
            A ModelResponse containing the model's text and any tool
            calls it requested.
        """
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        """Return a numeric embedding vector for text, for semantic
        (meaning-based) search -- see repo_index/semantic_index.py.

        A genuinely different capability from generate(), not a variant of
        it: this calls a separate, much smaller embedding model (not the
        coding model). Deliberately NOT abstract, unlike generate() --
        most backends (and every existing fake test model) have no reason
        to support this, the same way most Tool subclasses never override
        confirmation_message(). Defaults to raising, so a caller with no
        working embed() gets a clear, immediate failure rather than a
        silently wrong result; real support is opt-in per backend.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support embeddings."
        )
