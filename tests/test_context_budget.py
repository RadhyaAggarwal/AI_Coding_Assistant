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
    assert result.endswith("... (truncated)")
    assert result.startswith("x" * 4000)


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
