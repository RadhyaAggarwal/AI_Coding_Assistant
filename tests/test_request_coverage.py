from model_interface.base import ModelInterface, ModelResponse

from agent_controller.request_coverage import find_unaddressed_part

_COMPOUND_REQUEST = (
    "Which files import repo_index.indexer, and where is the function "
    "cap_observation actually called?"
)
_SIMPLE_REQUEST = "Where is the ToolRegistry class defined?"


class _FixedReplyModel(ModelInterface):
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, messages, tools=None):
        self.prompts.append(messages[-1].content)
        return ModelResponse(text=self.reply)


def test_skips_the_model_call_entirely_for_a_simple_request():
    """Most requests are a single part with nothing to miss — paying even
    the bounded cost of a verification call for those would be waste."""
    model = _FixedReplyModel("this should never be read")
    result = find_unaddressed_part(_SIMPLE_REQUEST, "X is defined in a.py.", model)
    assert result is None
    assert model.prompts == []  # the model was never even called


def test_returns_none_when_model_says_complete():
    model = _FixedReplyModel("COMPLETE")
    result = find_unaddressed_part(_COMPOUND_REQUEST, "X is defined in a.py.", model)
    assert result is None


def test_returns_none_for_complete_with_minor_padding():
    model = _FixedReplyModel("The answer is complete.")
    result = find_unaddressed_part(_COMPOUND_REQUEST, "X is defined in a.py.", model)
    assert result is None


def test_returns_description_when_model_flags_a_gap():
    model = _FixedReplyModel("You never said which files import Y.")
    result = find_unaddressed_part(_COMPOUND_REQUEST, "X is defined in a.py.", model)
    assert result == "You never said which files import Y."


def test_long_reply_mentioning_complete_word_is_not_treated_as_complete():
    long_reply = ("complete " + "details " * 10).strip()
    model = _FixedReplyModel(long_reply)
    result = find_unaddressed_part(_COMPOUND_REQUEST, "some answer", model)
    assert result == long_reply


def test_sends_request_and_answer_in_the_prompt():
    model = _FixedReplyModel("COMPLETE")
    find_unaddressed_part(_COMPOUND_REQUEST, "MY ANSWER TEXT", model)
    assert _COMPOUND_REQUEST in model.prompts[-1]
    assert "MY ANSWER TEXT" in model.prompts[-1]
