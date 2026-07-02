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
