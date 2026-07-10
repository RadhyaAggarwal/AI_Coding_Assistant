from agent_controller.conversation_store import load_conversation, save_conversation
from model_interface.base import Message


def test_round_trips_a_conversation(tmp_path):
    messages = [
        Message(role="system", content="SYS"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
    ]

    save_conversation(tmp_path, messages)
    loaded = load_conversation(tmp_path)

    assert loaded == messages


def test_load_returns_none_when_nothing_was_ever_saved(tmp_path):
    assert load_conversation(tmp_path) is None


def test_load_returns_none_on_corrupt_file_instead_of_raising(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "last_conversation.json").write_text("not valid json{{{", encoding="utf-8")

    assert load_conversation(tmp_path) is None


def test_save_creates_the_state_directory_if_missing(tmp_path):
    state_dir = tmp_path / "does" / "not" / "exist" / "yet"

    save_conversation(state_dir, [Message(role="user", content="hi")])

    assert load_conversation(state_dir) == [Message(role="user", content="hi")]


def test_saving_again_overwrites_the_previous_conversation(tmp_path):
    save_conversation(tmp_path, [Message(role="user", content="first")])
    save_conversation(tmp_path, [Message(role="user", content="second")])

    assert load_conversation(tmp_path) == [Message(role="user", content="second")]
