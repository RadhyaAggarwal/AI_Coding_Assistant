from tools.confirmation import prompt_confirm


def test_prompt_confirm_accepts_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert prompt_confirm("do it?") is True


def test_prompt_confirm_accepts_yes_full_word(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Yes")
    assert prompt_confirm("do it?") is True


def test_prompt_confirm_rejects_other_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "no")
    assert prompt_confirm("do it?") is False


def test_prompt_confirm_rejects_empty_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert prompt_confirm("do it?") is False


def test_prompt_confirm_treats_other_text_as_a_decline_reason(monkeypatch):
    """Anything typed that isn't y/yes/n/no/empty is a decline *with*
    feedback -- the live-observed gap this exists to close: a plain
    decline gave the model no information to act on, unlike a
    syntax_check rejection, which always includes a specific reason."""
    monkeypatch.setattr("builtins.input", lambda _: "there would be duplicate lines of code")
    result = prompt_confirm("do it?")
    assert result == "there would be duplicate lines of code"
    assert result is not True
    assert result is not False
