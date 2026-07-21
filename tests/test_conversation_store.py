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
    loaded_messages, loaded_already_called, loaded_call_results = load_conversation(tmp_path)

    assert loaded_messages == messages
    assert loaded_already_called == set()
    assert loaded_call_results == {}


def test_round_trips_already_called_alongside_messages(tmp_path):
    messages = [Message(role="user", content="hello")]
    already_called = {("read_file", '{"path": "a.py"}'), ("run_command", '{"command": "pytest"}')}

    save_conversation(tmp_path, messages, already_called)
    loaded_messages, loaded_already_called, _ = load_conversation(tmp_path)

    assert loaded_messages == messages
    assert loaded_already_called == already_called


def test_round_trips_call_results_alongside_already_called(tmp_path):
    messages = [Message(role="user", content="hello")]
    already_called = {("read_file", '{"path": "a.py"}')}
    call_results = {("read_file", '{"path": "a.py"}'): "def foo(): pass"}

    save_conversation(tmp_path, messages, already_called, call_results)
    _, _, loaded_call_results = load_conversation(tmp_path)

    assert loaded_call_results == call_results


def test_load_defaults_call_results_to_empty_on_older_pre_call_results_format(tmp_path):
    """Unlike a missing already_called key (treated as "nothing to
    continue" below), a file saved before call_results existed should
    still load fine -- losing it only means a duplicate rejection falls
    back to the old generic wording, not a wrong or unsafe outcome."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "last_conversation.json").write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "already_called": [["read_file", '{"path": "a.py"}']],
            }
        ),
        encoding="utf-8",
    )

    result = load_conversation(tmp_path)

    assert result is not None
    loaded_messages, loaded_already_called, loaded_call_results = result
    assert loaded_messages == [Message(role="user", content="hi")]
    assert loaded_already_called == {("read_file", '{"path": "a.py"}')}
    assert loaded_call_results == {}


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

    loaded_messages, _, _ = load_conversation(state_dir)
    assert loaded_messages == [Message(role="user", content="hi")]


def test_saving_again_overwrites_the_previous_conversation(tmp_path):
    save_conversation(tmp_path, [Message(role="user", content="first")])
    save_conversation(tmp_path, [Message(role="user", content="second")])

    loaded_messages, _, _ = load_conversation(tmp_path)
    assert loaded_messages == [Message(role="user", content="second")]
