"""Minimal agent loop for the vertical-slice milestone.

Ask the model whether it needs a tool, run the tool call it names, feed
the result back, and return a final answer. This proves the model /
tool / controller pieces connect end-to-end — it is intentionally not the
full understand -> plan -> act -> observe -> continue ReAct system yet.

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
from pathlib import Path

from model_interface.base import ModelInterface, ModelResponse, ToolCall
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


def run(user_request: str, model: ModelInterface, tools: ToolRegistry) -> str:
    conversation = f"{_load_system_prompt()}\n\nUser request: {user_request}"
    known_tool_names = {schema["function"]["name"] for schema in tools.schemas()}

    for attempt in range(1, _MAX_TOOL_ATTEMPTS + 1):
        response = model.generate(conversation, tools=tools.schemas())
        call = _resolve_tool_call(response, known_tool_names)

        if call is None:
            return response.text

        try:
            observation = tools.execute(call.name, call.arguments)
        except Exception as exc:
            if attempt == _MAX_TOOL_ATTEMPTS:
                return (
                    f"Could not complete the request: attempted to call "
                    f"'{call.name}' with {call.arguments}, which failed: {exc}"
                )
            conversation += (
                f"\n\nYou attempted: {call.name}({call.arguments})\n"
                f"That failed: {exc}\n"
                "Correct the call and try again, or answer directly if you can't."
            )
            continue

        conversation += (
            f"\n\nTool call: {call.name}({call.arguments})\n"
            f"Observation:\n{observation}\n\n"
            "Using the observation above, give the user a final answer."
        )
        # Don't offer tools on this call: the loop doesn't act on a second
        # tool call anyway, and offering one biases small models toward
        # emitting more tool-call JSON instead of a natural-language answer.
        final_response = model.generate(conversation)
        return final_response.text

    return "Could not complete the request."
