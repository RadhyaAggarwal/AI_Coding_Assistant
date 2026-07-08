"""Ollama backend for ModelInterface.

Talks to a local Ollama server's /api/chat endpoint over HTTP. This is the
only file in the project that knows Ollama's request/response shape.
Recovering a tool call the model attempted outside the structured
tool_calls field (see model_interface/tool_call_parsing.py) is handled by
agent_controller/, not here, so every backend benefits from it uniformly
instead of each adapter having to remember to do it.
"""
from typing import Any

import requests

from model_interface.base import Message, ModelInterface, ModelResponse, ToolCall


class OllamaAdapter(ModelInterface):
    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        request_timeout_seconds: float = 120,
        temperature: float | None = None,
    ):
        self._endpoint_url = endpoint_url.rstrip("/")
        self._model_name = model_name
        self._timeout = request_timeout_seconds
        self._temperature = temperature

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if self._temperature is not None:
            payload["options"] = {"temperature": self._temperature}

        response = requests.post(
            f"{self._endpoint_url}/api/chat",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        text = message.get("content", "")

        tool_calls = [
            ToolCall(
                name=call["function"]["name"],
                arguments=call["function"].get("arguments", {}),
            )
            for call in message.get("tool_calls", [])
        ]

        return ModelResponse(text=text, tool_calls=tool_calls, raw=data)
