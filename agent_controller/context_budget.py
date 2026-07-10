"""Minimal context budgeting for the conversation sent to the model.

Per the Build Plan Addendum (2.4): "simple truncation/prioritization" is
the right first version of a context manager — full compression/
summarization is deferred until this proves insufficient in practice.
Deliberately truncation, not summarization: summarizing would mean
asking this same model (already observed to fabricate when given too
little grounding — see find_symbol's history) to compress its own tool
outputs, adding both latency and a new fabrication surface for a problem
we haven't yet observed actually happening.

Two independent safeguards:
  - cap_observation(): a hard cap on any single tool result before it
    enters the conversation, regardless of whether that tool already
    self-limits (some do — search_code, run_command, find_symbol; some
    don't — read_file has no cap of its own). This is a backstop, not a
    replacement for per-tool limits.
  - trim_to_budget(): drops the oldest messages once the running total
    risks exceeding the model's context window, keeping the system
    prompt and the original first user message pinned, and never
    dropping a "tool" message without the "assistant" turn that
    requested it (so a kept tool observation is never left without the
    context of why it was fetched). Originally assumed everything after
    the first two messages was a clean, alternating (assistant, tool)*
    sequence -- true for a single request, but broken once
    agent_controller/loop.py's `transcript` parameter (see there) can
    seed a run with an *extended* conversation containing an earlier
    request's own "user" turn partway through, not just at the start.
    Rewritten to trim message-by-message from the oldest end instead of
    assuming rigid pairing, so a mid-history "user" turn from a prior
    --continue is treated as its own droppable-or-keepable unit rather
    than corrupting the pairing logic.

Token counts here are a character-based estimate (roughly 4 chars per
token for English text), not an exact tokenizer count — good enough for
budgeting headroom, not for precise limits.
"""
from model_interface.base import Message

_CHARS_PER_TOKEN_ESTIMATE = 4
_MAX_OBSERVATION_CHARS = 4000
_RESPONSE_HEADROOM_TOKENS = 1000  # leave room for the model's own reply


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def cap_observation(text: str) -> str:
    if len(text) <= _MAX_OBSERVATION_CHARS:
        return text
    return text[:_MAX_OBSERVATION_CHARS] + "\n... (truncated)"


def trim_to_budget(messages: list[Message], context_window_tokens: int) -> list[Message]:
    budget = context_window_tokens - _RESPONSE_HEADROOM_TOKENS
    total = sum(estimate_tokens(m.content) for m in messages)
    if total <= budget:
        return messages

    head = messages[:2]  # system, original first user message — always kept
    rest = messages[2:]

    running = sum(estimate_tokens(m.content) for m in head)
    kept: list[Message] = []
    i = len(rest) - 1
    while i >= 0:
        # A "tool" message is only ever kept together with the
        # "assistant" turn that requested it, walking back two at a time
        # in that shape. Anything else (an assistant turn with no tool
        # response, or a later "user" turn from a --continue follow-up)
        # is its own droppable-or-keepable unit, one at a time.
        if rest[i].role == "tool" and i > 0 and rest[i - 1].role == "assistant":
            chunk = rest[i - 1 : i + 1]
        else:
            chunk = rest[i : i + 1]
        cost = sum(estimate_tokens(m.content) for m in chunk)
        if running + cost > budget:
            break
        kept = chunk + kept
        running += cost
        i -= len(chunk)

    dropped = len(rest) - len(kept)
    if dropped == 0:
        return messages

    notice = Message(
        role="user",
        content=f"[{dropped} earlier message(s) omitted to stay within the context budget]",
    )
    return list(head) + [notice] + kept
