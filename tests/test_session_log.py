from agent_controller.session_log import (
    _MAX_FIELD_CHARS,
    _MAX_RETAINED_SESSIONS,
    append_session,
    read_sessions,
)


def test_round_trips_a_session(tmp_path):
    append_session(tmp_path, "summarize config.yaml", "It has model/project/state sections.", completed=True)

    entries = read_sessions(tmp_path)

    assert len(entries) == 1
    assert entries[0]["request"] == "summarize config.yaml"
    assert entries[0]["outcome"] == "It has model/project/state sections."
    assert entries[0]["completed"] is True
    assert isinstance(entries[0]["timestamp"], float)


def test_multiple_appends_accumulate_in_order(tmp_path):
    append_session(tmp_path, "first request", "first outcome", completed=True)
    append_session(tmp_path, "second request", "second outcome", completed=False)

    entries = read_sessions(tmp_path)

    assert [e["request"] for e in entries] == ["first request", "second request"]
    assert [e["completed"] for e in entries] == [True, False]


def test_read_sessions_returns_empty_list_when_nothing_logged(tmp_path):
    assert read_sessions(tmp_path) == []


def test_read_sessions_returns_empty_list_on_corrupt_file(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "session_log.json").write_text("not valid json{{{", encoding="utf-8")

    assert read_sessions(tmp_path) == []


def test_read_sessions_respects_limit(tmp_path):
    for i in range(5):
        append_session(tmp_path, f"request {i}", f"outcome {i}", completed=True)

    entries = read_sessions(tmp_path, limit=2)

    assert [e["request"] for e in entries] == ["request 3", "request 4"]


def test_long_request_and_outcome_are_truncated(tmp_path):
    long_text = "x" * (_MAX_FIELD_CHARS + 100)

    append_session(tmp_path, long_text, long_text, completed=True)

    entries = read_sessions(tmp_path)
    assert len(entries[0]["request"]) == _MAX_FIELD_CHARS + 3  # + "..."
    assert entries[0]["request"].endswith("...")
    assert len(entries[0]["outcome"]) == _MAX_FIELD_CHARS + 3
    assert entries[0]["outcome"].endswith("...")


def test_short_request_and_outcome_are_not_truncated(tmp_path):
    append_session(tmp_path, "short", "also short", completed=True)

    entries = read_sessions(tmp_path)
    assert entries[0]["request"] == "short"
    assert entries[0]["outcome"] == "also short"


def test_caps_total_retained_sessions(tmp_path):
    for i in range(_MAX_RETAINED_SESSIONS + 10):
        append_session(tmp_path, f"request {i}", "outcome", completed=True)

    entries = read_sessions(tmp_path)

    assert len(entries) == _MAX_RETAINED_SESSIONS
    # the oldest entries were dropped, the most recent ones kept
    assert entries[0]["request"] == "request 10"
    assert entries[-1]["request"] == f"request {_MAX_RETAINED_SESSIONS + 9}"
