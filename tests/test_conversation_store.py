import json

from agent_controller.conversation_store import load_conversation, save_conversation
from model_interface.base import Message


def test_round_trips_a_conversation(tmp_path):
    messages = [
        Message(role="system", content="SYS"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
    ]

    save_conversation(tmp_path, messages)
    loaded_messages, loaded_already_called = load_conversation(tmp_path)

    assert loaded_messages == messages
    assert loaded_already_called == set()


def test_round_trips_already_called_alongside_messages(tmp_path):
    messages = [Message(role="user", content="hello")]
    already_called = {("read_file", '{"path": "a.py"}'), ("run_command", '{"command": "pytest"}')}

    save_conversation(tmp_path, messages, already_called)
    loaded_messages, loaded_already_called = load_conversation(tmp_path)

    assert loaded_messages == messages
    assert loaded_already_called == already_called


def test_load_returns_none_when_nothing_was_ever_saved(tmp_path):
    assert load_conversation(tmp_path) is None


def test_load_returns_none_on_corrupt_file_instead_of_raising(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "last_conversation.json").write_text("not valid json{{{", encoding="utf-8")

    assert load_conversation(tmp_path) is None


def test_load_returns_none_on_older_pre_already_called_format(tmp_path):
    """A file saved by a previous version of this module was a bare
    list of messages, not {"messages": ..., "already_called": ...} --
    treated the same as "nothing there" rather than raising, consistent
    with how any other damaged/unreadable state file is handled here."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "last_conversation.json").write_text(
        json.dumps([{"role": "user", "content": "hi"}]), encoding="utf-8"
    )

    assert load_conversation(tmp_path) is None


def test_save_creates_the_state_directory_if_missing(tmp_path):
    state_dir = tmp_path / "does" / "not" / "exist" / "yet"

    save_conversation(state_dir, [Message(role="user", content="hi")])

    loaded_messages, _ = load_conversation(state_dir)
    assert loaded_messages == [Message(role="user", content="hi")]


def test_saving_again_overwrites_the_previous_conversation(tmp_path):
    save_conversation(tmp_path, [Message(role="user", content="first")])
    save_conversation(tmp_path, [Message(role="user", content="second")])

    loaded_messages, _ = load_conversation(tmp_path)
    assert loaded_messages == [Message(role="user", content="second")]
