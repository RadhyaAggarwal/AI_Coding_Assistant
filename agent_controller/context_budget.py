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
  - trim_to_budget(): drops the oldest assistant/tool exchanges (in
    pairs, so a tool observation is never kept without the assistant
    turn that requested it) once the running total risks exceeding the
    model's context window. The system and user messages are always
    kept.

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

    head = messages[:2]  # system, user — always kept
    rest = messages[2:]  # a sequence of (assistant, tool) pairs

    pairs = [rest[i : i + 2] for i in range(0, len(rest), 2)]

    running = sum(estimate_tokens(m.content) for m in head)
    kept_pairs: list[list[Message]] = []
    for pair in reversed(pairs):
        cost = sum(estimate_tokens(m.content) for m in pair)
        if running + cost > budget:
            break
        kept_pairs.append(pair)
        running += cost
    kept_pairs.reverse()

    dropped = len(pairs) - len(kept_pairs)
    if dropped == 0:
        return messages

    notice = Message(
        role="user",
        content=f"[{dropped} earlier tool exchange(s) omitted to stay within the context budget]",
    )
    trimmed = list(head) + [notice]
    for pair in kept_pairs:
        trimmed.extend(pair)
    return trimmed
