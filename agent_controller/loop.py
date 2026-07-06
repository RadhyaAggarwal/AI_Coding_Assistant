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
over long ReAct loops"), so this loop defends against six failure modes
rather than trusting the model to self-regulate:
  - Not using the backend's structured tool-calling field at all, just
    writing the JSON as plain text. _resolve_tool_calls() recovers that.
  - Planning more than one tool call in a single turn (e.g. a JSON array
    covering both parts of a compound request) and having all but the
    first silently discarded. Observed live: the model correctly planned
    find_importers + find_callers together, only the first ran, and the
    model then reported a false "no calls found" for the one that never
    executed instead of realizing it was never checked. _resolve_tool_calls()
    (plural) and the per-call loop below run every call from one turn,
    not just the first, bounded by _MAX_CALLS_PER_TURN.
  - Passing arguments that don't match the tool's schema, or naming a
    tool that fails outright. Fed back as a "tool" turn so the model can
    correct itself, instead of the loop crashing.
  - Calling the exact same tool with the exact same arguments again
    (getting stuck in a loop) instead of using the observation it
    already has. Detected and refused rather than trusted to the
    system prompt's "don't do that" instruction alone.
  - Answering a compound request without actually addressing every part
    of it — a model can produce a confident-sounding answer that's
    really a refusal, a guess, or drops a part it already investigated.
    find_unaddressed_part() (agent_controller/request_coverage.py) asks
    the model itself whether its own answer is complete (a short,
    one-time-per-request check — recognizing a non-answer or a semantic
    match with no shared vocabulary, e.g. "the RepoIndex class" needing
    find_symbol, isn't something keyword matching can do reliably), and
    gets one nudge to fix it before finishing.
  - Attempting a tool call that fails to parse at all (e.g. a docstring's
    unescaped quotes breaking a JSON string value it's embedded in) and
    having the leftover raw, broken-looking text silently returned to the
    user as if it were a genuine final answer — worse than no answer.
    looks_like_unparsed_tool_call() (model_interface/tool_call_parsing.py)
    distinguishes "no tool call was attempted" from "one was attempted
    and broke" by reusing the same brace-scan extraction already does; a
    detected broken attempt gets fed back as a nudge to fix the JSON
    instead of being handed to the user.
A hard step budget bounds all of the above combined, so a model that
never converges can't run forever.
"""
import json
from pathlib import Path

from agent_controller.context_budget import cap_observation, trim_to_budget
from agent_controller.request_coverage import find_unaddressed_part
from agent_controller.tool_router import route_tools
from model_interface.base import Message, ModelInterface, ModelResponse, ToolCall
from model_interface.tool_call_parsing import extract_tool_calls, looks_like_unparsed_tool_call
from tools.registry import ToolRegistry

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.md"
# Conservative default so existing callers (tests) don't need to pass this;
# main.py passes the real value from config.yaml (model.context_window_tokens).
_DEFAULT_CONTEXT_WINDOW_TOKENS = 8192
# 4 was enough for a single tool call; a real edit -> test -> fix -> retest
# cycle needs more room (read, edit, test-fail, edit-fix, test-pass, then a
# final answer attempt all count as steps). Each step is ~50-240s on this
# machine's CPU-only inference, so this is a real latency tradeoff, not a
# free increase — raise further only if a realistic task still runs out.
_MAX_STEPS = 6
# Observed live: a model can plan several tool calls in a single turn (a
# JSON array covering both parts of a compound request) — a real plan,
# not garbage, and worth executing in full within that one step (see
# _resolve_tool_calls). This bounds how many calls from one turn we'll
# trust and run, the same defensive-against-unreliable-output posture as
# the rest of this loop, in case a bad turn ever lists an implausible
# number of calls.
_MAX_CALLS_PER_TURN = 4


def _load_system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _resolve_tool_calls(response: ModelResponse, known_tool_names: set[str]) -> list[ToolCall]:
    if response.tool_calls:
        calls = list(response.tool_calls)
    else:
        calls = extract_tool_calls(response.text, known_tool_names)
    return calls[:_MAX_CALLS_PER_TURN]


def _assistant_turn_content(response: ModelResponse, calls: list[ToolCall]) -> str:
    # Preserve exactly what the model said when it said anything; only
    # synthesize a stand-in when native tool_calls left the text empty.
    if response.text:
        return response.text
    return json.dumps([{"name": call.name, "arguments": call.arguments} for call in calls])


def _call_key(call: ToolCall) -> tuple[str, str]:
    return (call.name, json.dumps(call.arguments, sort_keys=True))


def run(
    user_request: str,
    model: ModelInterface,
    tools: ToolRegistry,
    context_window_tokens: int = _DEFAULT_CONTEXT_WINDOW_TOKENS,
) -> str:
    messages = [
        Message(role="system", content=_load_system_prompt()),
        Message(role="user", content=user_request),
    ]
    known_tool_names = {schema["function"]["name"] for schema in tools.schemas()}
    always_keep_names = frozenset(tools.confirmation_required_names())
    already_called: set[tuple[str, str]] = set()
    coverage_nudge_used = False

    for _ in range(_MAX_STEPS):
        sendable = trim_to_budget(messages, context_window_tokens)
        query_text = " ".join(m.content for m in sendable if m.role != "system")
        offered_tools = route_tools(query_text, tools.schemas(), always_keep_names=always_keep_names)
        response = model.generate(sendable, tools=offered_tools)
        calls = _resolve_tool_calls(response, known_tool_names)

        if not calls:
            if looks_like_unparsed_tool_call(response.text, known_tool_names):
                messages.append(Message(role="assistant", content=response.text))
                messages.append(
                    Message(
                        role="user",
                        content=(
                            "That didn't parse as a valid tool call — check for "
                            "unescaped quotes or other JSON syntax errors (e.g. a "
                            "docstring's quotes breaking a string value) and no "
                            "tool ran. Fix the JSON and try again, or answer in "
                            "plain text without attempting a tool call."
                        ),
                    )
                )
                continue
            if not coverage_nudge_used:
                unaddressed = find_unaddressed_part(user_request, response.text, model)
                if unaddressed is not None:
                    coverage_nudge_used = True
                    messages.append(Message(role="assistant", content=response.text))
                    messages.append(
                        Message(
                            role="user",
                            content=(
                                f"Your answer isn't complete: {unaddressed} Please "
                                "address that (using a tool if needed) before "
                                "finishing."
                            ),
                        )
                    )
                    continue
            return response.text

        messages.append(
            Message(role="assistant", content=_assistant_turn_content(response, calls))
        )

        for call in calls:
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
            messages.append(Message(role="tool", content=cap_observation(observation)))

    # Step budget exhausted. Ask once more, without tool access, so the
    # model synthesizes an answer from whatever it gathered rather than
    # the loop just giving up with nothing. If it still tries to call a
    # tool anyway, don't leak that raw attempt to the user as if it were
    # a real answer.
    final_response = model.generate(trim_to_budget(messages, context_window_tokens))
    if _resolve_tool_calls(final_response, known_tool_names) or looks_like_unparsed_tool_call(
        final_response.text, known_tool_names
    ):
        return f"Could not complete the request within {_MAX_STEPS} steps."
    return final_response.text
