from agent_controller.change_summary import changed_files
from model_interface.base import Message


def test_detects_a_successful_edit_file_call():
    transcript = [
        Message(role="system", content="sys"),
        Message(role="user", content="fix it"),
        Message(role="tool", content="Edited scratch_scoring.py."),
    ]
    assert changed_files(transcript) == ["scratch_scoring.py"]


def test_detects_a_successful_create_file_call_for_a_new_file():
    transcript = [Message(role="tool", content="Created new_module.py.")]
    assert changed_files(transcript) == ["new_module.py"]


def test_detects_a_successful_create_file_call_that_replaced_an_existing_file():
    transcript = [Message(role="tool", content="Replaced existing.py.")]
    assert changed_files(transcript) == ["existing.py"]


def test_ignores_a_failed_edit_file_call():
    transcript = [
        Message(
            role="tool",
            content="Error: 'search' text was not found in 'sample.py'. No changes made.",
        ),
    ]
    assert changed_files(transcript) == []


def test_ignores_unrelated_tool_results():
    transcript = [
        Message(role="tool", content="No symbol named 'foo' found."),
        Message(role="tool", content="Exit code: 0\nsome output"),
    ]
    assert changed_files(transcript) == []


def test_ignores_non_tool_messages_even_if_they_mention_a_fix():
    """The exact live case this module exists to route around: don't
    trust what the model's own answer text claims -- only real tool
    results count."""
    transcript = [
        Message(role="assistant", content="I proposed a fix by modifying the division step..."),
    ]
    assert changed_files(transcript) == []


def test_deduplicates_repeated_edits_to_the_same_file_keeping_first_position():
    transcript = [
        Message(role="tool", content="Edited a.py."),
        Message(role="tool", content="Edited b.py."),
        Message(role="tool", content="Edited a.py."),
    ]
    assert changed_files(transcript) == ["a.py", "b.py"]


def test_empty_transcript_returns_no_changes():
    assert changed_files([]) == []


def test_extracts_the_real_path_when_an_advisory_note_is_attached():
    """Reproduces the exact live bug: edit_file's result carries a note
    as further lines after the confirmation ("Edited {path}.\nNote:
    ..."), and matching against the WHOLE content instead of just the
    first line swept the note's own text into what should have been a
    clean path, corrupting the printed summary."""
    transcript = [
        Message(
            role="tool",
            content=(
                "Edited core/views.py.\nNote: line 25: unreachable code -- "
                "follows an unconditional return/raise/break/continue "
                "earlier in the same block."
            ),
        ),
    ]
    assert changed_files(transcript) == ["core/views.py"]


def test_extracts_the_real_path_when_a_create_file_note_is_attached():
    transcript = [
        Message(
            role="tool",
            content=(
                "Created scratch.py.\nNote: line 2: 'foo' is used but "
                "doesn't appear to be imported or defined anywhere in "
                "this file -- this may cause a NameError at runtime."
            ),
        ),
    ]
    assert changed_files(transcript) == ["scratch.py"]


def test_deduplicates_correctly_when_only_one_of_two_edits_carries_a_note():
    transcript = [
        Message(role="tool", content="Edited a.py.\nNote: line 1: unreachable code."),
        Message(role="tool", content="Edited a.py."),
    ]
    assert changed_files(transcript) == ["a.py"]
