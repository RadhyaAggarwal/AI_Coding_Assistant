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

from model_interface.base import (
    Message,
    ModelInterface,
    ModelResponse,
    ModelUnavailableError,
    ToolCall,
)


class OllamaAdapter(ModelInterface):
    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        request_timeout_seconds: float = 120,
        temperature: float | None = None,
        health_check_timeout_seconds: float = 5,
    ):
        self._endpoint_url = endpoint_url.rstrip("/")
        self._model_name = model_name
        self._timeout = request_timeout_seconds
        self._temperature = temperature
        self._health_check_timeout = health_check_timeout_seconds

    def _check_alive(self) -> None:
        """Fail fast if Ollama isn't responding, rather than blocking a full
        request_timeout_seconds on a call that may never come back.

        A stuck Ollama process (observed live: still unresponsive 10+
        minutes after a request was abandoned client-side) can silently
        queue every subsequent request behind it. A lightweight endpoint
        that doesn't touch the model itself (listing installed models,
        rather than generating from one) should stay responsive even while
        a generate call is stuck, letting this distinguish "slow" from
        "unresponsive" in a few seconds instead of the full timeout.

        Also checks the response's status code, not just whether the
        connection itself failed -- a broken tunnel/proxy hop can return a
        connection-level success with a 404/502/503 body, which used to
        pass this check silently and only surface later as an ugly raw
        traceback from the real request.
        """
        try:
            response = requests.get(
                f"{self._endpoint_url}/api/tags",
                timeout=self._health_check_timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise ModelUnavailableError(
                f"Ollama at {self._endpoint_url} did not respond to a "
                f"health check within {self._health_check_timeout}s. It "
                f"looks unavailable or stuck on an earlier request -- "
                f"try restarting the Ollama process before retrying."
            ) from exc

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self._check_alive()

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if self._temperature is not None:
            payload["options"] = {"temperature": self._temperature}

        try:
            response = requests.post(
                f"{self._endpoint_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            # Covers a timeout/connection failure (Ollama itself stuck)
            # and an HTTP error status (a broken tunnel/proxy hop) alike --
            # our request is always well-formed, so any failure talking to
            # Ollama is realistically an availability problem on the other
            # end, not a bug in what we sent. Observed live: an unhandled
            # HTTPError (a 404, then later a 503, from a flaky tunnel) hit
            # main.py as a raw traceback instead of the clean message this
            # exception type is supposed to produce.
            raise ModelUnavailableError(
                f"Ollama at {self._endpoint_url} did not respond within "
                f"{self._timeout}s (passed its health check moments "
                f"earlier, so it may have gotten stuck mid-request -- try "
                f"restarting the Ollama process before retrying)."
            ) from exc
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
