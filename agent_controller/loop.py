"""Bounded multi-step agent loop: understand -> act -> observe -> continue.

Ask the model whether it needs a tool, run whatever it calls, feed the
result back, and let it decide whether to call another tool or answer.
This is a real (if simple) ReAct loop now — not the full planner from
the Build Plan, but genuinely multi-step, unlike the earlier one-tool
version. The conversation is proper role-separated messages (system,
user, assistant, tool), which is what makes extending to multiple steps
straightforward instead of a rewrite.

Local models are unreliable in ways that matter a lot more once a loop
can run several steps (Build Plan Addendum 2.2: "degrading plan quality
over long ReAct loops"), so this loop defends against three failure
modes rather than trusting the model to self-regulate:
  - Not using the backend's structured tool-calling field at all, just
    writing the JSON as plain text. _resolve_tool_call() recovers that.
  - Passing arguments that don't match the tool's schema, or naming a
    tool that fails outright. Fed back as a "tool" turn so the model can
    correct itself, instead of the loop crashing.
  - Calling the exact same tool with the exact same arguments again
    (getting stuck in a loop) instead of using the observation it
    already has. Detected and refused rather than trusted to the
    system prompt's "don't do that" instruction alone.
A hard step budget bounds all of the above combined, so a model that
never converges can't run forever.
"""
import json
from pathlib import Path

from model_interface.base import Message, ModelInterface, ModelResponse, ToolCall
from model_interface.tool_call_parsing import extract_tool_call
from tools.registry import ToolRegistry

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.md"
_MAX_STEPS = 4


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


def _call_key(call: ToolCall) -> tuple[str, str]:
    return (call.name, json.dumps(call.arguments, sort_keys=True))


def run(user_request: str, model: ModelInterface, tools: ToolRegistry) -> str:
    messages = [
        Message(role="system", content=_load_system_prompt()),
        Message(role="user", content=user_request),
    ]
    known_tool_names = {schema["function"]["name"] for schema in tools.schemas()}
    already_called: set[tuple[str, str]] = set()

    for _ in range(_MAX_STEPS):
        response = model.generate(messages, tools=tools.schemas())
        call = _resolve_tool_call(response, known_tool_names)

        if call is None:
            return response.text

        messages.append(
            Message(role="assistant", content=_assistant_turn_content(response, call))
        )

        if _call_key(call) in already_called:
            messages.append(
                Message(
                    role="tool",
                    content=(
                        "You already called this exact tool with these exact "
                        "arguments. Use the observation you already have to "
                        "answer, or call a different tool."
                    ),
                )
            )
            continue

        try:
            observation = tools.execute(call.name, call.arguments)
        except Exception as exc:
            messages.append(
                Message(
                    role="tool",
                    content=f"Error: {exc}. Correct the call and try again, or answer directly if you can't.",
                )
            )
            continue

        already_called.add(_call_key(call))
        messages.append(Message(role="tool", content=observation))

    # Step budget exhausted. Ask once more, without tool access, so the
    # model synthesizes an answer from whatever it gathered rather than
    # the loop just giving up with nothing. If it still tries to call a
    # tool anyway, don't leak that raw attempt to the user as if it were
    # a real answer.
    final_response = model.generate(messages)
    if _resolve_tool_call(final_response, known_tool_names) is not None:
        return f"Could not complete the request within {_MAX_STEPS} steps."
    return final_response.text
