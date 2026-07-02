"""Minimal agent loop for the vertical-slice milestone.

Ask the model whether it needs a tool, run the tool call it names, feed
the result back, and return a final answer. This proves the model /
tool / controller pieces connect end-to-end — it is intentionally not the
full understand -> plan -> act -> observe -> continue ReAct system yet.

The conversation is built as proper role-separated messages (system,
user, assistant, tool) rather than one flattened string, so it reads as
a coherent history to the model and so a future multi-step ReAct loop can
extend this same history instead of needing a different representation.

Local models are unreliable about tool calls in two distinct ways (Build
Plan Addendum 2.2), both handled here rather than left to each backend:
  - They may not use the backend's structured tool-calling field at all,
    instead just writing the JSON as plain text. _resolve_tool_call()
    recovers that via model_interface.tool_call_parsing so every
    ModelInterface implementation benefits, not just one adapter.
  - Once a call is recovered, its arguments may not match the tool's
    schema (missing/wrong-typed fields). We tell the model what went
    wrong and give it one more chance before giving up.
"""
import json
from pathlib import Path

from model_interface.base import Message, ModelInterface, ModelResponse, ToolCall
from model_interface.tool_call_parsing import extract_tool_call
from tools.registry import ToolRegistry

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.md"
_MAX_TOOL_ATTEMPTS = 2


def _load_system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _resolve_tool_call(response: ModelResponse, known_tool_names: set[str]) -> ToolCall | None:
    if response.tool_calls:
        return response.tool_calls[0]
    return extract_tool_call(response.text, known_tool_names)


def _assistant_turn_content(response: ModelResponse, call: ToolCall) -> str:
    # Preserve exactly what the model said when it said anything; only
    # synthesize a stand-in when native tool_calls left the text empty.
    return response.text or json.dumps({"name": call.name, "arguments": call.arguments})


def run(user_request: str, model: ModelInterface, tools: ToolRegistry) -> str:
    messages = [
        Message(role="system", content=_load_system_prompt()),
        Message(role="user", content=user_request),
    ]
    known_tool_names = {schema["function"]["name"] for schema in tools.schemas()}

    for attempt in range(1, _MAX_TOOL_ATTEMPTS + 1):
        response = model.generate(messages, tools=tools.schemas())
        call = _resolve_tool_call(response, known_tool_names)

        if call is None:
            return response.text

        messages.append(
            Message(role="assistant", content=_assistant_turn_content(response, call))
        )

        try:
            observation = tools.execute(call.name, call.arguments)
        except Exception as exc:
            if attempt == _MAX_TOOL_ATTEMPTS:
                return (
                    f"Could not complete the request: attempted to call "
                    f"'{call.name}' with {call.arguments}, which failed: {exc}"
                )
            messages.append(
                Message(
                    role="tool",
                    content=f"Error: {exc}. Correct the call and try again.",
                )
            )
            continue

        messages.append(Message(role="tool", content=observation))
        # Don't offer tools on this call: the loop doesn't act on a second
        # tool call anyway, and offering one biases small models toward
        # emitting more tool-call JSON instead of a natural-language answer.
        final_response = model.generate(messages)
        return final_response.text

    return "Could not complete the request."
