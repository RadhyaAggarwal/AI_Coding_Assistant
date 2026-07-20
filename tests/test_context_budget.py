from agent_controller.context_budget import cap_observation, estimate_tokens, trim_to_budget
from model_interface.base import Message


def test_estimate_tokens_roughly_four_chars_per_token():
    assert estimate_tokens("a" * 400) == 100


def test_estimate_tokens_minimum_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("hi") == 1


def test_cap_observation_leaves_short_text_untouched():
    text = "short result"
    assert cap_observation(text) == text


def test_cap_observation_truncates_long_text():
    text = "x" * 5000
    result = cap_observation(text)
    assert len(result) < len(text)
    assert result.startswith("x" * 2500)
    assert result.endswith("x" * 1500)
    assert "1000 chars omitted" in result


def test_cap_observation_preserves_the_tail_not_just_the_head():
    """Reproduces a real live-found bug: a Python traceback's exception
    type and message are always the last line -- head-only truncation
    silently discarded exactly that for a run_command traceback over the
    cap, leaving the model with plausible-looking upper stack frames and
    no way to see the actual NameError. The tail must survive."""
    traceback_text = (
        "Traceback (most recent call last):\n"
        + "  File \"module.py\", line 1, in <module>\n" * 300
        + "NameError: name 'include' is not defined"
    )
    result = cap_observation(traceback_text)
    assert result.endswith("NameError: name 'include' is not defined")


def test_trim_to_budget_returns_unchanged_when_within_budget():
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="call"),
        Message(role="tool", content="observation"),
    ]
    result = trim_to_budget(messages, context_window_tokens=8192)
    assert result == messages


def test_trim_to_budget_drops_oldest_pairs_when_over_budget():
    chunk = "x" * 400  # ~100 tokens
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="request"),
    ]
    for i in range(5):
        messages.append(Message(role="assistant", content=f"call {i}: {chunk}"))
        messages.append(Message(role="tool", content=f"result {i}: {chunk}"))

    # budget (context - 1000 headroom) = 250: fits head + 1 pair, not 2
    result = trim_to_budget(messages, context_window_tokens=1250)

    assert result[0] == messages[0]
    assert result[1] == messages[1]
    assert "omitted" in result[2].content
    assert result[-2].content.startswith("call 4:")  # most recent pair kept
    assert result[-1].content.startswith("result 4:")


def test_trim_to_budget_never_orphans_a_tool_message():
    chunk = "x" * 400
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="request"),
    ]
    for i in range(5):
        messages.append(Message(role="assistant", content=f"call {i}: {chunk}"))
        messages.append(Message(role="tool", content=f"result {i}: {chunk}"))

    result = trim_to_budget(messages, context_window_tokens=1250)

    roles = [m.role for m in result]
    for i, role in enumerate(roles):
        if role == "tool":
            assert roles[i - 1] == "assistant"


def test_trim_to_budget_handles_a_continuations_mid_history_user_turn():
    """A --continue follow-up seeds run() with an *extended* conversation
    containing an earlier request's own "user" turn partway through, not
    just at the start -- a shape the original (assistant, tool)*-only
    pairing assumption didn't anticipate. This reproduces that shape
    directly (not via the full loop) to isolate the trimming logic."""
    chunk = "x" * 400
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="first request"),
        Message(role="assistant", content=f"call 0: {chunk}"),
        Message(role="tool", content=f"result 0: {chunk}"),
        Message(role="assistant", content="first answer"),
        Message(role="user", content="second request (a --continue follow-up)"),
        Message(role="assistant", content=f"call 1: {chunk}"),
        Message(role="tool", content=f"result 1: {chunk}"),
    ]

    result = trim_to_budget(messages, context_window_tokens=1250)

    roles = [m.role for m in result]
    for i, role in enumerate(roles):
        if role == "tool":
            assert roles[i - 1] == "assistant"
    # the newest turn -- the --continue follow-up and what it actually
    # asked for -- must survive; it's what's currently being worked on
    assert result[-2].content.startswith("call 1:")
    assert result[-1].content.startswith("result 1:")
    assert any(
        m.content == "second request (a --continue follow-up)" for m in result
    )
