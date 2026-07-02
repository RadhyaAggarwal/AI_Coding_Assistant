"""Ollama backend for ModelInterface.

Talks to a local Ollama server's /api/chat endpoint over HTTP. This is the
only file in the project that knows Ollama's request/response shape.
"""
from typing import Any

import requests

from model_interface.base import ModelInterface, ModelResponse, ToolCall


class OllamaAdapter(ModelInterface):
    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        request_timeout_seconds: float = 120,
    ):
        self._endpoint_url = endpoint_url.rstrip("/")
        self._model_name = model_name
        self._timeout = request_timeout_seconds

    def generate(
        self,
        prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

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
