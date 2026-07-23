"""End-to-end tests for agent_controller.loop.run against fake models.

Covers the real failure mode observed against Ollama (tool call written as
plain text instead of using the structured tool_calls field), the bounded
multi-step chaining added afterward, and the defenses that make chaining
safe (repeated-call refusal, a hard step budget). Fake models branch on
message-list shape (length/roles) rather than string content, since the
conversation is a proper role-separated history, not a flattened string.
"""
import json

from model_interface.base import Message, ModelInterface, ModelResponse, ModelUnavailableError
from tools.create_file import CreateFileTool
from tools.edit_file import EditFileTool
from tools.list_directory import ListDirectoryTool
from tools.read_file import ReadFileTool
from tools.registry import ToolRegistry
from tools.run_command import RunCommandTool
from tools.search_code import SearchCodeTool

from agent_controller.loop import (
    _MAX_CALLS_PER_TURN,
    _MAX_STEPS,
    _MAX_WASTED_STEPS,
    _load_system_prompt,
    is_incomplete_answer,
    run,
)


def test_load_system_prompt_strips_the_leading_html_comment():
    """The STATUS/ownership comment at the top of system_prompt.md is
    developer-facing (see CLAUDE.md's division-of-labor rule), not
    model-facing -- it must never reach the real system message sent to
    the model."""
    prompt = _load_system_prompt()
    assert "<!--" not in prompt
    assert "-->" not in prompt
    assert "STATUS" not in prompt


def test_load_system_prompt_keeps_the_real_instructions():
    prompt = _load_system_prompt()
    assert "Understand the request" in prompt


def _call_text(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments})


class HugeObservationModel(ModelInterface):
    """Calls a tool that returns a very large observation, then
    answers — verifies the loop caps it before it enters the
    conversation (context_budget.cap_observation), regardless of
    whether the tool itself self-limits (read_file doesn't)."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        if len(messages) == 2:
            return ModelResponse(text=_call_text("read_file", {"path": "huge.txt"}))
        return ModelResponse(text="Got it.")


def test_huge_tool_observation_gets_capped(tmp_path):
    (tmp_path / "huge.txt").write_text("x" * 10000, encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = HugeObservationModel()

    answer = run("Summarize huge.txt", model, tools)

    assert answer == "Got it."
    tool_message = model.calls[-1][-1]
    assert tool_message.role == "tool"
    assert len(tool_message.content) < 10000
    assert "truncated" in tool_message.content


class RepeatedLargeReadModel(ModelInterface):
    """Reads several different large files in sequence, then answers —
    used to verify that once accumulated history exceeds a tight budget,
    the message list actually sent to the model stops growing without
    bound, rather than the loop just accumulating forever."""

    def __init__(self):
        self.sent_message_counts: list[int] = []
        self._step = 0

    def generate(self, messages, tools=None):
        self.sent_message_counts.append(len(messages))
        if self._step < 5:
            path = f"file{self._step}.txt"
            self._step += 1
            return ModelResponse(text=_call_text("read_file", {"path": path}))
        return ModelResponse(text="Done.")


def test_trim_to_budget_keeps_sent_history_bounded_under_tight_budget(tmp_path):
    for i in range(5):
        (tmp_path / f"file{i}.txt").write_text("x" * 4000, encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = RepeatedLargeReadModel()

    answer = run("Read all the files", model, tools, context_window_tokens=2100)

    assert answer == "Done."
    # Without trimming, the raw history would keep growing every call
    # (2, 4, 6, 8, 10, 12). With a tight budget, later calls should not
    # reach that untrimmed size.
    untrimmed_final_size = 2 + 2 * 5
    assert model.sent_message_counts[-1] < untrimmed_final_size


class TextOnlyToolCallModel(ModelInterface):
    """Mimics the observed qwen2.5-coder:7b behavior: the tool call is
    written as plain JSON text in the response content, never populating
    the native tool_calls field."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))  # snapshot: loop mutates this list in place
        if len(messages) == 2:  # system + user: first turn, no history yet
            return ModelResponse(
                text='{"name": "read_file", "arguments": {"path": "sample.txt"}}'
            )
        return ModelResponse(text="FINAL ANSWER USING OBSERVATION")


def test_recovers_text_only_tool_call_and_executes_it(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = TextOnlyToolCallModel()

    answer = run("Summarize sample.txt", model, tools)

    assert answer == "FINAL ANSWER USING OBSERVATION"
    assert len(model.calls) == 2
    final_call_messages = model.calls[1]
    assert [m.role for m in final_call_messages] == ["system", "user", "assistant", "tool"]
    assert final_call_messages[-1].content == "sample contents"  # observation fed back verbatim


class FailThenSucceedModel(ModelInterface):
    """First attempt omits the required 'path' argument; second attempt
    (after seeing the failure) supplies it correctly."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))  # snapshot: loop mutates this list in place
        if len(messages) == 2:
            return ModelResponse(text='{"name": "read_file", "arguments": {}}')
        if len(messages) == 4:
            return ModelResponse(
                text='{"name": "read_file", "arguments": {"path": "sample.txt"}}'
            )
        return ModelResponse(text="FINAL ANSWER")


def test_retries_after_validation_failure(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = FailThenSucceedModel()

    answer = run("Summarize sample.txt", model, tools)

    assert answer == "FINAL ANSWER"
    assert len(model.calls) == 3
    # the retry should have seen the validation error as a tool-role message
    retry_messages = model.calls[1]
    assert retry_messages[-1].role == "tool"
    assert "Missing required argument" in retry_messages[-1].content


class AlwaysInvalidModel(ModelInterface):
    """Always calls read_file missing the required 'path' argument, but
    varies an extra field each time so every attempt is a distinct (if
    still invalid) call -- exercises genuine _MAX_STEPS exhaustion
    specifically, not the separate wasted-turn cap that an exact repeat
    would now correctly trip (see RepeatsFailingEditModel for that)."""

    def __init__(self):
        self.call_count = 0

    def generate(self, messages, tools=None):
        self.call_count += 1
        return ModelResponse(text=f'{{"name": "read_file", "arguments": {{"attempt": {self.call_count}}}}}')


class RecoversWithPlainLanguageAfterNudgeModel(ModelInterface):
    """Behaves exactly like AlwaysInvalidModel through the whole step
    budget and the first final-synthesis attempt (still reaching for a
    tool out of habit even though none are offered) -- but responds with
    a real plain-language partial summary once told explicitly that no
    tools are available. Verifies the explicit nudge actually recovers a
    usable answer instead of the loop giving up with a generic message."""

    def __init__(self):
        self.call_count = 0

    def generate(self, messages, tools=None):
        self.call_count += 1
        if self.call_count <= _MAX_STEPS + 1:
            return ModelResponse(
                text=f'{{"name": "read_file", "arguments": {{"attempt": {self.call_count}}}}}'
            )
        return ModelResponse(text="Here's what I found so far: nothing conclusive yet.")


def test_gives_up_after_max_steps(tmp_path):
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = AlwaysInvalidModel()

    answer = run("Summarize sample.txt", model, tools)

    assert f"within {_MAX_STEPS} steps" in answer
    assert is_incomplete_answer(answer)
    # _MAX_STEPS failed attempts, plus the final no-tools synthesis call,
    # plus one more explicit nudge to stop trying tools and summarize --
    # both still fail to produce a clean answer (guarded, not leaked raw)
    assert model.call_count == _MAX_STEPS + 2


def test_recovers_with_plain_language_summary_after_explicit_nudge(tmp_path):
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = RecoversWithPlainLanguageAfterNudgeModel()

    answer = run("Summarize sample.txt", model, tools)

    assert answer == "Here's what I found so far: nothing conclusive yet."
    assert not is_incomplete_answer(answer)
    # _MAX_STEPS failed attempts + the first final synthesis attempt
    # (still broken) + the one explicit nudge, which succeeds this time.
    assert model.call_count == _MAX_STEPS + 2


class ImmediateAnswerModel(ModelInterface):
    """Answers directly in plain text on the first call, no tool use --
    used to isolate transcript behavior from tool-call mechanics."""

    def __init__(self, answer_text="ANSWER"):
        self.calls: list[list[Message]] = []
        self._answer_text = answer_text

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        return ModelResponse(text=self._answer_text)


def test_transcript_empty_list_behaves_like_a_fresh_conversation(tmp_path):
    tools = ToolRegistry()
    model = ImmediateAnswerModel("ANSWER")
    transcript: list[Message] = []

    answer = run("Summarize sample.txt", model, tools, transcript=transcript)

    assert answer == "ANSWER"
    first_call_messages = model.calls[0]
    assert [m.role for m in first_call_messages] == ["system", "user"]
    # after the call, transcript should hold the full conversation,
    # including the final answer recorded as an assistant turn -- ready
    # to be saved and passed back in on a later --continue call.
    assert [m.role for m in transcript] == ["system", "user", "assistant"]
    assert transcript[-1].content == "ANSWER"


def test_transcript_with_prior_history_is_used_as_the_starting_point(tmp_path):
    """Reproduces what --continue is meant to do: a saved conversation
    from an earlier call gets used as real starting context, not
    discarded in favor of a fresh system+user exchange."""
    tools = ToolRegistry()
    model = ImmediateAnswerModel("SECOND ANSWER")
    prior = [
        Message(role="system", content="SYS PROMPT"),
        Message(role="user", content="first question"),
        Message(role="assistant", content="first answer"),
    ]
    transcript = list(prior)

    answer = run("second question", model, tools, transcript=transcript)

    assert answer == "SECOND ANSWER"
    first_call_messages = model.calls[0]
    # the prior history is sent as-is, with the new user turn appended
    # after it -- not replaced by a fresh system+user pair.
    assert [m.content for m in first_call_messages] == [
        "SYS PROMPT", "first question", "first answer", "second question",
    ]
    assert [m.content for m in transcript] == [
        "SYS PROMPT", "first question", "first answer", "second question", "SECOND ANSWER",
    ]


class UnavailableOnFirstCallModel(ModelInterface):
    """Raises ModelUnavailableError on the very first generate() call --
    reproduces the original --continue-after-failure bug, where nothing
    had been synced to transcript yet when the exception propagated."""

    def generate(self, messages, tools=None):
        raise ModelUnavailableError("simulated timeout")


def test_transcript_synced_before_propagating_model_unavailable_on_first_call(tmp_path):
    tools = ToolRegistry()
    model = UnavailableOnFirstCallModel()
    transcript: list[Message] = []
    already_called: set[tuple[str, str]] = set()

    try:
        run("fix the bug", model, tools, transcript=transcript, already_called=already_called)
        assert False, "expected ModelUnavailableError to propagate"
    except ModelUnavailableError:
        pass

    # the pending, unanswered request is what got saved -- not left empty
    # -- so a later --continue resumes the actual failed attempt instead
    # of falling back to whatever unrelated conversation was saved before.
    assert [m.role for m in transcript] == ["system", "user"]
    assert transcript[-1].content == "fix the bug"


class UnavailableAfterOneToolCallModel(ModelInterface):
    """Succeeds on step one (a real tool call), then fails on step two --
    verifies real progress made before the failure is preserved, not just
    the original request."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(text=_call_text("list_directory", {"path": "."}))
        raise ModelUnavailableError("simulated timeout")


def test_transcript_synced_before_propagating_model_unavailable_after_a_tool_call(tmp_path):
    tools = ToolRegistry()
    tools.register(ListDirectoryTool(tmp_path))
    model = UnavailableAfterOneToolCallModel()
    transcript: list[Message] = []
    already_called: set[tuple[str, str]] = set()

    try:
        run("look around then explain", model, tools, transcript=transcript, already_called=already_called)
        assert False, "expected ModelUnavailableError to propagate"
    except ModelUnavailableError:
        pass

    assert [m.role for m in transcript] == ["system", "user", "assistant", "tool"]
    # the real tool call that already succeeded is remembered too, so a
    # resumed --continue won't redundantly repeat it.
    assert already_called == {("list_directory", json.dumps({"path": "."}))}


class UnavailableOnCoverageCheckModel(ModelInterface):
    """Answers in plain text (no tool call) to a compound request, then
    raises ModelUnavailableError on the follow-up coverage-check call
    (agent_controller/request_coverage.py's own model.generate) -- a
    second, separate call site that also needs to sync before raising."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(text="Here is part of the answer.")
        raise ModelUnavailableError("simulated timeout")


def test_transcript_synced_before_propagating_model_unavailable_from_coverage_check(tmp_path):
    tools = ToolRegistry()
    model = UnavailableOnCoverageCheckModel()
    transcript: list[Message] = []

    try:
        run(
            "Which files import repo_index.indexer, and where is the function "
            "cap_observation actually called?",
            model,
            tools,
            transcript=transcript,
        )
        assert False, "expected ModelUnavailableError to propagate"
    except ModelUnavailableError:
        pass

    # the model's real (if incomplete) answer is preserved, not discarded,
    # even though the coverage check that would have nudged it never
    # got to finish.
    assert [m.role for m in transcript] == ["system", "user", "assistant"]
    assert transcript[-1].content == "Here is part of the answer."


class BatchesTwoMutatingEditsInOneTurnModel(ModelInterface):
    """Plans two edit_file calls together as a single JSON array in one
    turn -- reproduces the real live bug: a model batched four calls in
    one turn (edit views.py, edit urls.py to reference what the first
    edit was supposed to add, edit a template, run a check), all decided
    before any of them had actually run. The first edit failed; the
    second ran anyway, wiring a reference to something that was never
    added -- exactly the broken state a real check later caught."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(
                text=json.dumps(
                    [
                        {
                            "name": "edit_file",
                            "arguments": {"path": "a.py", "search": "x = 1", "replace": "x = 2"},
                        },
                        {
                            "name": "edit_file",
                            "arguments": {"path": "a.py", "search": "y = 1", "replace": "y = 2"},
                        },
                    ]
                )
            )
        return ModelResponse(text="DONE")


def test_batch_execution_stops_after_the_first_mutating_call(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 1\n", encoding="utf-8")
    tools = ToolRegistry(confirm=lambda description: True)
    tools.register(EditFileTool(tmp_path))
    model = BatchesTwoMutatingEditsInOneTurnModel()

    run("Update a.py", model, tools)

    # only the first of the two batched edits actually ran -- the second
    # must not execute blind to whether the first one succeeded
    content = (tmp_path / "a.py").read_text(encoding="utf-8")
    assert "x = 2" in content
    assert "y = 1" in content

    # proves the second call was never *executed* at all in that first
    # turn (not just that its effect happened to not show up) -- the
    # model's next turn only ever saw one tool result, not two
    second_turn_messages = model.calls[1]
    tool_messages_seen = [m for m in second_turn_messages if m.role == "tool"]
    assert len(tool_messages_seen) == 1


class BatchesFailingEditThenAnotherMutatingCallModel(ModelInterface):
    """The exact real bug shape, not just the success-path variant above:
    the first batched call *fails* (a wrong search string, same as the
    real "# Add your views here." case), and a second, different
    mutating call is queued right alongside it in the same turn --
    reproduces core/urls.py getting wired to a function core/views.py
    never actually gained, because the second edit ran regardless of the
    first one's real, failed outcome."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(
                text=json.dumps(
                    [
                        {
                            "name": "edit_file",
                            "arguments": {"path": "a.py", "search": "nonexistent text", "replace": "x = 2"},
                        },
                        {
                            "name": "edit_file",
                            "arguments": {"path": "b.py", "search": "y = 1", "replace": "y = 2"},
                        },
                    ]
                )
            )
        return ModelResponse(text="DONE")


def test_batch_execution_stops_after_a_mutating_call_fails(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")
    tools = ToolRegistry(confirm=lambda description: True)
    tools.register(EditFileTool(tmp_path))
    model = BatchesFailingEditThenAnotherMutatingCallModel()

    run("Update both files", model, tools)

    # a.py's edit failed (search text didn't exist there) -- b.py's edit,
    # queued right alongside it in the same batch, must not have executed
    # blind to that failure, the same way core/urls.py shouldn't have been
    # wired to a function core/views.py never actually gained
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "y = 1\n"

    second_turn_messages = model.calls[1]
    tool_messages_seen = [m for m in second_turn_messages if m.role == "tool"]
    assert len(tool_messages_seen) == 1
    assert "not found" in tool_messages_seen[0].content


class MultiStepModel(ModelInterface):
    """Calls two different tools in sequence before answering — exercises
    the new multi-step chaining capability."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(text='{"name": "search_code", "arguments": {"query": "login"}}')
        if len(tool_messages) == 1:
            return ModelResponse(text='{"name": "read_file", "arguments": {"path": "auth.py"}}')
        return ModelResponse(text="FINAL ANSWER FROM TWO TOOLS")


def test_chains_multiple_different_tool_calls(tmp_path):
    (tmp_path / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(SearchCodeTool(tmp_path))
    tools.register(ReadFileTool(tmp_path))
    model = MultiStepModel()

    answer = run("Find the login function and explain it", model, tools)

    assert answer == "FINAL ANSWER FROM TWO TOOLS"
    assert len(model.calls) == 3
    final_call_messages = model.calls[-1]
    assert [m.role for m in final_call_messages] == [
        "system", "user", "assistant", "tool", "assistant", "tool",
    ]


class RepeatsCallModel(ModelInterface):
    """Calls read_file, then tries the exact same call again instead of
    answering — the loop should refuse the repeat, not re-run the tool or
    loop forever."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) <= 1:
            return ModelResponse(text='{"name": "read_file", "arguments": {"path": "sample.txt"}}')
        return ModelResponse(text="FINAL ANSWER AFTER REPEAT WARNING")


def test_refuses_identical_repeated_tool_call(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = RepeatsCallModel()

    answer = run("Summarize sample.txt", model, tools)

    assert answer == "FINAL ANSWER AFTER REPEAT WARNING"
    assert len(model.calls) == 3
    # the repeat should have been refused with a warning, not re-executed
    # (which would have put "sample contents" there again instead)
    third_call_messages = model.calls[2]
    assert third_call_messages[-1].role == "tool"
    assert "already called this exact tool" in third_call_messages[-1].content


class RepeatsRunCommandBackToBackModel(ModelInterface):
    """Proposes the exact same run_command call twice in a row with
    nothing else happening in between -- reproduces the real live gap
    where run_command's blanket dedup_exempt let this straight through
    to a second full confirmation prompt for a command that had just
    failed, with nothing having changed (scratch_cache.py's
    temp_cache_test.py incident)."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) < 2:
            return ModelResponse(text=_call_text("run_command", {"command": "python nonexistent.py"}))
        return ModelResponse(text="FINAL ANSWER")


def test_run_command_back_to_back_identical_repeat_is_blocked(tmp_path):
    """The redesigned dedup_exempt behavior: run_command is no longer
    exempt from every duplicate check, only from the general call_history
    one -- an exact repeat of the single most-recently-executed call,
    with nothing else having happened in between, is still blocked."""
    tools = ToolRegistry(confirm=lambda description: True)
    tools.register(RunCommandTool(tmp_path))
    model = RepeatsRunCommandBackToBackModel()

    answer = run("Run the script", model, tools)

    assert answer == "FINAL ANSWER"
    third_call_messages = model.calls[2]
    rejection = third_call_messages[-1]
    assert rejection.role == "tool"
    assert "already called this exact tool" in rejection.content


class InterleavedDuplicatesModel(ModelInterface):
    """5 distinct real read_file calls, interleaved with 2 duplicate
    repeats of the very first one, then a final text answer -- 8 total
    generate() calls. Reproduces the live failure that motivated exempting
    duplicate-only turns from the primary step budget: a request that did
    real, correct work still failed with "could not complete within 6
    steps" because repeated (rejected) calls counted against the budget
    just like real ones did."""

    _SEQUENCE = ["a.txt", "a.txt", "b.txt", "a.txt", "c.txt", "d.txt", "e.txt"]

    def __init__(self):
        self.calls: list[list[Message]] = []
        self._step = 0

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        if self._step < len(self._SEQUENCE):
            path = self._SEQUENCE[self._step]
            self._step += 1
            return ModelResponse(text=_call_text("read_file", {"path": path}))
        return ModelResponse(text="FINAL ANSWER AFTER DUPLICATES")


def test_duplicate_only_turns_dont_consume_the_real_step_budget(tmp_path):
    for name in ("a", "b", "c", "d", "e"):
        (tmp_path / f"{name}.txt").write_text(f"{name} contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = InterleavedDuplicatesModel()

    answer = run("Investigate the project's files", model, tools)

    # 5 real reads (a, b, c, d, e) + 2 duplicate-only turns (both repeats
    # of a.txt) + 1 final text answer = 8 generate() calls -- more than
    # _MAX_STEPS (6), which would have failed before this change since
    # every for-loop iteration consumed the budget regardless of whether
    # it was a genuine duplicate rejection.
    assert answer == "FINAL ANSWER AFTER DUPLICATES"
    assert len(model.calls) == 8


class ForeverRepeatsCallModel(ModelInterface):
    """Never produces anything new -- always attempts the exact same
    read_file call, even after being told (via the dedup tool message)
    that it's a repeat. Verifies the loop still fails safely, bounded by
    _MAX_WASTED_STEPS, instead of running unbounded now that
    duplicate-only turns no longer consume the primary step budget."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        return ModelResponse(text=_call_text("read_file", {"path": "sample.txt"}))


def test_gives_up_when_stuck_repeating_the_same_call(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = ForeverRepeatsCallModel()

    answer = run("Investigate sample.txt", model, tools)

    assert "kept repeating" in answer
    assert is_incomplete_answer(answer)
    # 1 genuine first call + _MAX_WASTED_STEPS duplicate-only turns before
    # giving up, plus the final no-tools synthesis attempt, plus one more
    # explicit nudge to stop trying tools and summarize instead -- both
    # still fail to produce a clean answer here.
    assert len(model.calls) == 1 + _MAX_WASTED_STEPS + 2


def test_reports_when_skipping_a_duplicate_call(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    reported = []
    tools = ToolRegistry(report=reported.append)
    tools.register(ReadFileTool(tmp_path))
    model = RepeatsCallModel()

    run("Summarize sample.txt", model, tools)

    assert any("duplicate" in message.lower() for message in reported)


class RepeatsFailingEditModel(ModelInterface):
    """Keeps retrying the exact same edit_file call, which fails
    identically every time (the search text doesn't exist in the file).
    Reproduces the live failure this test targets: a call that never
    succeeds used to be invisible to the dedup guard (which only ever
    recorded successes), so every identical retry burned real step
    budget instead of being caught as a repeat."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        return ModelResponse(
            text=_call_text(
                "edit_file",
                {"path": "sample.txt", "search": "nonexistent text", "replace": "x"},
            )
        )


def test_dedups_an_identical_call_even_when_it_fails(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry(confirm=lambda description: True)
    tools.register(EditFileTool(tmp_path))
    model = RepeatsFailingEditModel()

    answer = run("Fix sample.txt", model, tools)

    assert "kept repeating" in answer
    assert is_incomplete_answer(answer)
    # 1 real failing attempt (still counts as real progress -- an error
    # is new information) + _MAX_WASTED_STEPS duplicate-only turns (each
    # identical retry now correctly deduped instead of burning real
    # budget) + the final no-tools synthesis attempt + one explicit
    # nudge.
    assert len(model.calls) == 1 + _MAX_WASTED_STEPS + 2


def test_duplicate_rejection_includes_the_real_cached_result(tmp_path):
    """A deduped repeat must hand the model back the real content the
    first call actually returned, not just tell it "use the observation
    you already have" -- live-observed that phrasing alone isn't enough,
    see the loop.py comment at the dedup-rejection site."""
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = RepeatsCallModel()

    run("Summarize sample.txt", model, tools)

    third_call_messages = model.calls[2]
    rejection = third_call_messages[-1]
    assert rejection.role == "tool"
    assert "sample contents" in rejection.content


def test_duplicate_rejection_of_a_failed_call_includes_the_real_error(tmp_path):
    """The same guarantee as above, for a call that fails identically
    every time (edit_file's search text never matching) rather than one
    that succeeds -- the real error message should carry over too, not
    just the fact that it was already tried."""
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry(confirm=lambda description: True)
    tools.register(EditFileTool(tmp_path))
    model = RepeatsFailingEditModel()

    run("Fix sample.txt", model, tools)

    third_call_messages = model.calls[2]
    rejection = third_call_messages[-1]
    assert rejection.role == "tool"
    assert "'search' text was not found" in rejection.content


class RerunsVerificationAfterFixModel(ModelInterface):
    """Reproduces the exact live failure: run a command, fix a bug via
    edit_file, then try to re-run the *same* command to verify the fix.
    That second call must actually re-execute, not get refused as
    "already called" -- the file changed in between, so the dedup
    guard's assumption (identical arguments always give an identical
    result) no longer holds."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(
                text=_call_text("run_command", {"command": "python -c \"print('checking')\""})
            )
        if len(tool_messages) == 1:
            return ModelResponse(
                text=_call_text("edit_file", {"path": "sample.py", "search": "return 1", "replace": "return 2"})
            )
        if len(tool_messages) == 2:
            return ModelResponse(
                text=_call_text("run_command", {"command": "python -c \"print('checking')\""})
            )
        return ModelResponse(text="Verified the fix.")


def test_allows_rerunning_identical_command_after_a_successful_edit(tmp_path):
    (tmp_path / "sample.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    tools = ToolRegistry(confirm=lambda description: True)
    tools.register(RunCommandTool(tmp_path))
    tools.register(EditFileTool(tmp_path))
    model = RerunsVerificationAfterFixModel()

    answer = run("Fix the bug in sample.py and verify it with a check", model, tools)

    assert answer == "Verified the fix."
    assert len(model.calls) == 4
    # the second run_command call (after the edit) must have actually
    # re-executed, not been refused as "already called"
    final_tool_message = model.calls[3][-1]
    assert final_tool_message.role == "tool"
    assert "already called" not in final_tool_message.content
    assert "checking" in final_tool_message.content


def test_seeded_already_called_does_not_block_a_rerun_after_a_fresh_edit(tmp_path):
    """The seed represents a call made in a *prior* --continue turn (e.g.
    a run_command check that ran before this turn started). It must not
    defeat the existing clear-on-confirmation-gated-success rule: once
    edit_file succeeds again in *this* turn, the file may have changed,
    so re-running that seeded command must still actually execute --
    exactly the same guarantee test_allows_rerunning_identical_command_
    after_a_successful_edit already covers within a single turn, now
    also holding across a turn boundary."""
    (tmp_path / "sample.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    tools = ToolRegistry(confirm=lambda description: True)
    tools.register(RunCommandTool(tmp_path))
    tools.register(EditFileTool(tmp_path))
    model = RerunsVerificationAfterFixModel()
    seeded_already_called = {
        ("run_command", json.dumps({"command": "python -c \"print('checking')\""}, sort_keys=True))
    }

    answer = run(
        "Fix the bug in sample.py and verify it with a check",
        model,
        tools,
        already_called=seeded_already_called,
    )

    assert answer == "Verified the fix."
    final_tool_message = model.calls[3][-1]
    assert final_tool_message.role == "tool"
    assert "already called" not in final_tool_message.content
    assert "checking" in final_tool_message.content


class RerunsSameCommandWithNothingElseHappeningModel(ModelInterface):
    """Reproduces the exact live case: a human explicitly asks to re-run
    a command that already succeeded, with nothing else having changed
    -- no edit_file in between, nothing to invalidate the dedup guard's
    usual clear-on-success rule. Unlike RerunsVerificationAfterFixModel,
    there's no edit step here at all."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(
                text=_call_text("run_command", {"command": "python -c \"print('checking')\""})
            )
        return ModelResponse(text="Verified the fix.")


def test_run_command_is_exempt_from_dedup_even_with_nothing_else_changed(tmp_path):
    """run_command must not be blocked by a seeded prior success even
    when nothing (no edit_file, no other call) happened in between --
    unlike edit_file/create_file, a repeat can still be legitimately
    wanted (a human just re-verifying), and confirmation already gates
    every invocation regardless of past outcome."""
    tools = ToolRegistry(confirm=lambda description: True)
    tools.register(RunCommandTool(tmp_path))
    model = RerunsSameCommandWithNothingElseHappeningModel()
    seeded_already_called = {
        ("run_command", json.dumps({"command": "python -c \"print('checking')\""}, sort_keys=True))
    }

    answer = run(
        "Please run the check again to double check",
        model,
        tools,
        already_called=seeded_already_called,
    )

    assert answer == "Verified the fix."
    first_tool_message = model.calls[1][-1]
    assert first_tool_message.role == "tool"
    assert "already called" not in first_tool_message.content
    assert "checking" in first_tool_message.content


class RepeatsSuccessfulEditModel(ModelInterface):
    """Unlike run_command, edit_file keeps no dedup exemption: an
    identical repeat after a real success would just fail on its own
    (the 'search' text it needs is already gone), so there's nothing
    useful about letting it re-reach the confirmation prompt."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        return ModelResponse(
            text=_call_text("edit_file", {"path": "sample.py", "search": "return 1", "replace": "return 2"})
        )


def test_edit_file_keeps_no_dedup_exemption_after_a_seeded_success(tmp_path):
    (tmp_path / "sample.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    reported = []
    tools = ToolRegistry(confirm=lambda description: True, report=reported.append)
    tools.register(EditFileTool(tmp_path))
    model = RepeatsSuccessfulEditModel()
    seeded_already_called = {
        ("edit_file", json.dumps({"path": "sample.py", "search": "return 1", "replace": "return 2"}, sort_keys=True))
    }

    run("Apply that same fix again", model, tools, already_called=seeded_already_called)

    assert any("duplicate" in message.lower() for message in reported)


class RepeatsReadAfterAFailedCommandModel(ModelInterface):
    """Reproduces the exact live bug: read a file, then run a command
    that accomplishes nothing (a bad path/typo -- returns an error
    *string*, never raises), then try the exact same read again. The
    read must be caught as a duplicate -- the failed command shouldn't
    have cleared anything, since nothing about it guarantees project
    state actually changed."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(text=_call_text("read_file", {"path": "sample.txt"}))
        if len(tool_messages) == 1:
            return ModelResponse(
                text=_call_text("run_command", {"command": "this-is-not-a-real-command"})
            )
        return ModelResponse(text=_call_text("read_file", {"path": "sample.txt"}))


def test_a_failed_run_command_does_not_clear_other_tools_dedup_memory(tmp_path):
    (tmp_path / "sample.txt").write_text("contents", encoding="utf-8")
    reported = []
    tools = ToolRegistry(confirm=lambda description: True, report=reported.append)
    tools.register(ReadFileTool(tmp_path))
    tools.register(RunCommandTool(tmp_path))
    model = RepeatsReadAfterAFailedCommandModel()

    run("Investigate sample.txt", model, tools)

    assert any("duplicate" in message.lower() for message in reported)


class RepeatsASeededCallModel(ModelInterface):
    """Always attempts the exact same read_file call -- the one already
    seeded into already_called, simulating a call made in a prior
    --continue turn with nothing in between to invalidate it."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        return ModelResponse(text=_call_text("read_file", {"path": "sample.txt"}))


def test_seeded_already_called_blocks_a_stale_repeat_from_a_prior_turn(tmp_path):
    (tmp_path / "sample.txt").write_text("contents", encoding="utf-8")
    reported = []
    tools = ToolRegistry(report=reported.append)
    tools.register(ReadFileTool(tmp_path))
    model = RepeatsASeededCallModel()
    seeded_already_called = {("read_file", json.dumps({"path": "sample.txt"}, sort_keys=True))}

    run("Look at sample.txt again", model, tools, already_called=seeded_already_called)

    assert any("duplicate" in message.lower() for message in reported)


def test_already_called_is_synced_back_to_the_caller_after_run(tmp_path):
    """Mirrors how `transcript` is mutated in place (see
    test_transcript_empty_list_behaves_like_a_fresh_conversation) so a
    caller can persist it and pass it back into the next --continue
    call, the same way main.py does."""
    (tmp_path / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(SearchCodeTool(tmp_path))
    tools.register(ReadFileTool(tmp_path))
    model = MultiStepModel()
    already_called: set[tuple[str, str]] = set()

    run("Find the login function and explain it", model, tools, already_called=already_called)

    assert already_called == {
        ("search_code", json.dumps({"query": "login"}, sort_keys=True)),
        ("read_file", json.dumps({"path": "auth.py"}, sort_keys=True)),
    }


def test_seeded_call_results_surfaces_real_content_from_a_prior_turn(tmp_path):
    """Reproduces the actual live bug this feature fixes: a real read
    that succeeded in an earlier --continue turn (seeded here the same
    way already_called itself gets seeded) must be handed back verbatim
    on a later duplicate, not just referenced. Without call_results, this
    is exactly the scenario where an explicit human "re-read this file,
    don't trust what you said earlier" instruction couldn't get through."""
    (tmp_path / "sample.txt").write_text("the real file content", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = RepeatsASeededCallModel()
    key = ("read_file", json.dumps({"path": "sample.txt"}, sort_keys=True))
    seeded_already_called = {key}
    seeded_call_results = {key: "the real file content"}

    run(
        "Look at sample.txt again",
        model,
        tools,
        already_called=seeded_already_called,
        call_results=seeded_call_results,
    )

    second_call_messages = model.calls[1]
    rejection = second_call_messages[-1]
    assert rejection.role == "tool"
    assert "the real file content" in rejection.content


def test_call_results_is_synced_back_to_the_caller_after_run(tmp_path):
    """Mirrors test_already_called_is_synced_back_to_the_caller_after_run
    -- same in-place-mutation contract, so main.py can persist it and
    pass it back into the next --continue call alongside already_called."""
    (tmp_path / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(SearchCodeTool(tmp_path))
    tools.register(ReadFileTool(tmp_path))
    model = MultiStepModel()
    call_results: dict[tuple[str, str], str] = {}

    run("Find the login function and explain it", model, tools, call_results=call_results)

    assert call_results[("read_file", json.dumps({"path": "auth.py"}, sort_keys=True))] == (
        "def login():\n    pass\n"
    )


def test_call_results_cleared_when_a_mutating_call_succeeds(tmp_path):
    """The cached-result companion to test_seeded_already_called_does_not_
    block_a_rerun_after_a_fresh_edit -- a stale cached read from before a
    successful edit_file/create_file call must not survive it, since
    project state may have changed underneath it. Shares call_history's
    exact clear() so this can't silently drift out of sync with it."""
    (tmp_path / "sample.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    tools = ToolRegistry(confirm=lambda description: True)
    tools.register(ReadFileTool(tmp_path))
    tools.register(EditFileTool(tmp_path))
    model = RerunsVerificationAfterFixReadModel()
    stale_key = ("read_file", json.dumps({"path": "sample.py"}, sort_keys=True))
    already_called: set[tuple[str, str]] = {stale_key}
    call_results: dict[tuple[str, str], str] = {stale_key: "def f():\n    return 1\n"}

    run(
        "Fix the bug in sample.py and confirm by reading it again",
        model,
        tools,
        already_called=already_called,
        call_results=call_results,
    )

    # the model re-reads the file after fixing it (same key as the stale
    # seed, since the arguments are identical), so the cache ends up
    # holding that key again -- but the clear must have genuinely
    # happened in between, or this would still hold the stale pre-edit
    # content instead of being overwritten with the real post-edit read.
    assert call_results[stale_key] == "def f():\n    return 2\n"


class RerunsVerificationAfterFixReadModel(ModelInterface):
    """Like RerunsVerificationAfterFixModel, but re-verifies with
    read_file (not dedup_exempt) instead of run_command (which is
    dedup_exempt and so wouldn't actually exercise the always_mutates
    clearing path this test targets)."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(
                text=_call_text("edit_file", {"path": "sample.py", "search": "return 1", "replace": "return 2"})
            )
        if len(tool_messages) == 1:
            return ModelResponse(text=_call_text("read_file", {"path": "sample.py"}))
        return ModelResponse(text="Verified the fix.")


class EditVerifyFixVerifyModel(ModelInterface):
    """Simulates a realistic fix cycle: create a buggy file (via
    create_file, since edit_file no longer creates anything), verify by
    reading it, fix it with a targeted edit_file call, verify again, then
    answer. 4 tool-call turns plus a final answer turn — exercises the
    headroom _MAX_STEPS=6 was raised to support (4 was only enough for a
    single tool call)."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        step = len([m for m in messages if m.role == "tool"])
        if step == 0:
            return ModelResponse(
                text=_call_text(
                    "create_file",
                    {"path": "calc.py", "content": "def add(a, b):\n    return a - b\n"},
                )
            )
        if step == 1:
            return ModelResponse(text=_call_text("read_file", {"path": "calc.py"}))
        if step == 2:
            return ModelResponse(
                text=_call_text(
                    "edit_file",
                    {"path": "calc.py", "search": "return a - b", "replace": "return a + b"},
                )
            )
        if step == 3:
            return ModelResponse(text=_call_text("read_file", {"path": "calc.py"}))
        return ModelResponse(text="Fixed: add() now returns a + b.")


def test_completes_realistic_edit_verify_fix_cycle_within_budget(tmp_path):
    tools = ToolRegistry(confirm=lambda description: True)  # edit_file/create_file require confirmation
    tools.register(EditFileTool(tmp_path))
    tools.register(CreateFileTool(tmp_path))
    tools.register(ReadFileTool(tmp_path))
    model = EditVerifyFixVerifyModel()

    answer = run("Fix the add function in calc.py", model, tools)

    assert answer == "Fixed: add() now returns a + b."
    assert len(model.calls) == 5  # 4 tool-call turns + 1 final answer turn
    assert "return a + b" in (tmp_path / "calc.py").read_text(encoding="utf-8")


class RecordingToolsModel(ModelInterface):
    """Records which tool schemas were actually offered on the first
    generate() call, to verify the router narrowed them rather than
    always offering the full registered set."""

    def __init__(self):
        self.offered_tool_names: list[list[str]] = []

    def generate(self, messages, tools=None):
        names = [t["function"]["name"] for t in tools] if tools else []
        self.offered_tool_names.append(names)
        return ModelResponse(text="Done.")


def test_tool_router_narrows_offered_tools_for_specific_query(tmp_path):
    # Mirrors the real 11-tool registry (main.py's build_tool_registry) —
    # a 5-tool registry sharing generic "file"/"directory" vocabulary
    # across most descriptions doesn't leave enough room to demonstrate
    # narrowing; the structurally-distinct tools below don't share that
    # vocabulary with a plain "list files" query.
    from tools.find_callers import FindCallersTool
    from tools.find_importers import FindImportersTool
    from tools.find_symbol import FindSymbolTool
    from tools.html_overview import HtmlOverviewTool
    from tools.repo_overview import RepoOverviewTool

    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    tools.register(ListDirectoryTool(tmp_path))
    tools.register(SearchCodeTool(tmp_path))
    tools.register(RunCommandTool(tmp_path))
    tools.register(EditFileTool(tmp_path))
    tools.register(CreateFileTool(tmp_path))
    tools.register(RepoOverviewTool(tmp_path))
    tools.register(FindSymbolTool(tmp_path))
    tools.register(HtmlOverviewTool(tmp_path))
    tools.register(FindImportersTool(tmp_path))
    tools.register(FindCallersTool(tmp_path))
    model = RecordingToolsModel()

    run("List the files in the current directory", model, tools)

    offered = model.offered_tool_names[0]
    assert "list_directory" in offered
    assert len(offered) < 11  # narrowed from the full 11 registered


class TwoStepModel(ModelInterface):
    """Calls a tool on step 1, then answers on step 2 -- used to check
    that orientation-tool protection applies on the first step only, not
    every step (which would erode the router's narrowing benefit
    broadly, a fix already rejected once for being too wide)."""

    def __init__(self):
        self.offered_tool_names: list[list[str]] = []

    def generate(self, messages, tools=None):
        names = [t["function"]["name"] for t in tools] if tools else []
        self.offered_tool_names.append(names)
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(
                text=_call_text("find_symbol", {"name": "generate_receipt_total"})
            )
        return ModelResponse(text="Done.")


def test_orientation_tools_offered_only_on_the_first_step(tmp_path):
    """Reproduces the live finding: list_directory scored 0 against a
    vague, filename-blind bug report and never got offered at all, even
    though a directory listing (real filenames like scratch_receipt.py)
    would have been the cheapest possible lead. Verifies it's force-
    included on the first step, when nothing is known yet, but not on
    later steps once real investigation is underway."""
    from tools.find_callers import FindCallersTool
    from tools.find_importers import FindImportersTool
    from tools.find_symbol import FindSymbolTool
    from tools.html_overview import HtmlOverviewTool
    from tools.repo_overview import RepoOverviewTool

    (tmp_path / "sample.py").write_text("def generate_receipt_total():\n    pass\n", encoding="utf-8")

    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    tools.register(ListDirectoryTool(tmp_path))
    tools.register(SearchCodeTool(tmp_path))
    tools.register(RunCommandTool(tmp_path))
    tools.register(EditFileTool(tmp_path))
    tools.register(CreateFileTool(tmp_path))
    tools.register(RepoOverviewTool(tmp_path))
    tools.register(FindSymbolTool(tmp_path))
    tools.register(HtmlOverviewTool(tmp_path))
    tools.register(FindImportersTool(tmp_path))
    tools.register(FindCallersTool(tmp_path))
    model = TwoStepModel()

    run(
        "There's a bug in how receipt totals are calculated somewhere in "
        "the order/pricing code -- can you find and fix it, and verify "
        "the fix?",
        model,
        tools,
    )

    first_step_offered = model.offered_tool_names[0]
    assert "list_directory" in first_step_offered
    assert "repo_overview" in first_step_offered

    second_step_offered = model.offered_tool_names[1]
    assert "list_directory" not in second_step_offered
    assert "repo_overview" not in second_step_offered


class SkipsSecondPartThenCorrectsModel(ModelInterface):
    """Reproduces the exact live failure request_coverage.py was built to
    fix: calls find_importers, then answers prematurely without ever
    calling find_callers for the request's second part. request_coverage
    now asks the model itself whether the answer is complete — that
    verification call is always a single bare user-role message (no
    system prompt, no history), which is how this fake distinguishes it
    from the loop's normal decision calls. Only after that check reports
    the gap does it go back and call find_callers."""

    def __init__(self):
        self.calls: list[list[Message]] = []
        self._verify_call_count = 0

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))

        if len(messages) == 1:  # request_coverage's verification call
            self._verify_call_count += 1
            if self._verify_call_count == 1:
                return ModelResponse(text="You never checked where cap_observation is actually called.")
            return ModelResponse(text="COMPLETE")

        tool_messages = [m for m in messages if m.role == "tool"]
        user_messages = [m for m in messages if m.role == "user"]

        if len(tool_messages) == 0:
            return ModelResponse(
                text=_call_text("find_importers", {"module_name": "repo_index.indexer"})
            )
        if len(tool_messages) == 1 and len(user_messages) == 1:
            return ModelResponse(text="repo_index.indexer is imported by a.py.")
        if len(tool_messages) == 1 and len(user_messages) == 2:
            return ModelResponse(text=_call_text("find_callers", {"name": "cap_observation"}))
        return ModelResponse(text="Imported by a.py; cap_observation is called in b.py.")


def test_coverage_check_nudges_model_to_address_skipped_part(tmp_path):
    from tools.find_callers import FindCallersTool
    from tools.find_importers import FindImportersTool

    tools = ToolRegistry()
    tools.register(FindImportersTool(tmp_path))
    tools.register(FindCallersTool(tmp_path))
    model = SkipsSecondPartThenCorrectsModel()

    answer = run(
        "Which files import repo_index.indexer, and where is the function "
        "cap_observation actually called?",
        model,
        tools,
    )

    assert answer == "Imported by a.py; cap_observation is called in b.py."
    # find_callers must have actually been invoked at some point, not just
    # mentioned in the final answer text
    assert any(
        m.role == "assistant" and "find_callers" in m.content
        for messages in model.calls
        for m in messages
    )


class TwoCallsPlannedInOneTurnModel(ModelInterface):
    """Reproduces the exact live shape found via instrumentation: the
    model plans both parts of a compound request as a single JSON array
    in one turn, instead of one call per turn."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(
                text=json.dumps(
                    [
                        {"name": "find_importers", "arguments": {"module_name": "agent_controller.tool_router"}},
                        {"name": "find_callers", "arguments": {"name": "snippet_at"}},
                    ]
                )
            )
        return ModelResponse(text="Imported by loop.py; snippet_at is called in find_symbol.py.")


def test_executes_every_tool_call_planned_in_a_single_turn(tmp_path):
    from tools.find_callers import FindCallersTool
    from tools.find_importers import FindImportersTool

    tools = ToolRegistry()
    tools.register(FindImportersTool(tmp_path))
    tools.register(FindCallersTool(tmp_path))
    model = TwoCallsPlannedInOneTurnModel()

    answer = run(
        "Which files import agent_controller.tool_router, and where is snippet_at called?",
        model,
        tools,
    )

    assert answer == "Imported by loop.py; snippet_at is called in find_symbol.py."
    # Only 2 model calls needed: one turn planning both calls, one to
    # answer -- both tool calls must have executed within that single
    # planning turn rather than the second being silently dropped.
    assert len(model.calls) == 2
    final_messages = model.calls[-1]
    tool_messages = [m for m in final_messages if m.role == "tool"]
    assert len(tool_messages) == 2


class ThreeDistinctCallsInOneTurnModel(ModelInterface):
    """Not a 2-call special case: plans three *different* tools in one
    turn, under the per-turn cap, to prove the loop-level execution isn't
    hardcoded to the exact pair observed live."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(
                text=json.dumps(
                    [
                        {"name": "search_code", "arguments": {"query": "login"}},
                        {"name": "read_file", "arguments": {"path": "auth.py"}},
                        {"name": "list_directory", "arguments": {"path": "."}},
                    ]
                )
            )
        return ModelResponse(text="FINAL ANSWER FROM THREE TOOLS")


def test_executes_three_distinct_calls_planned_in_a_single_turn(tmp_path):
    (tmp_path / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(SearchCodeTool(tmp_path))
    tools.register(ReadFileTool(tmp_path))
    tools.register(ListDirectoryTool(tmp_path))
    model = ThreeDistinctCallsInOneTurnModel()

    answer = run("Find the login function, read it, and list the directory", model, tools)

    assert answer == "FINAL ANSWER FROM THREE TOOLS"
    assert len(model.calls) == 2  # one planning turn, one answer turn
    final_messages = model.calls[-1]
    tool_messages = [m for m in final_messages if m.role == "tool"]
    assert len(tool_messages) == 3


class TooManyCallsInOneTurnModel(ModelInterface):
    """Plans an implausible number of calls in a single turn -- more
    likely garbage than a real plan; the loop should bound how many it
    trusts and runs from one turn."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(
                text=json.dumps(
                    [{"name": "read_file", "arguments": {"path": f"f{i}.txt"}} for i in range(6)]
                )
            )
        return ModelResponse(text="Done.")


def test_caps_how_many_calls_from_one_turn_are_executed(tmp_path):
    for i in range(6):
        (tmp_path / f"f{i}.txt").write_text(f"content{i}", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = TooManyCallsInOneTurnModel()

    run("Read all six files", model, tools)

    final_messages = model.calls[-1]
    tool_messages = [m for m in final_messages if m.role == "tool"]
    assert len(tool_messages) == _MAX_CALLS_PER_TURN


class MalformedThenValidAnswerModel(ModelInterface):
    """Reproduces the exact live failure: first turn emits a tool-call
    attempt broken by an embedded docstring's own unescaped quotes;
    second turn (after being nudged) gives a genuine plain-text answer."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return ModelResponse(
                text=(
                    '{"name": "edit_file", "arguments": {"path": "a.py", "search": "", '
                    '"replace": "def f():\\n    """docstring"""\\n    return 1"}}'
                )
            )
        return ModelResponse(text="Fixed the function.")


def test_malformed_tool_call_attempt_is_not_returned_as_final_answer(tmp_path):
    tools = ToolRegistry(confirm=lambda description: True)
    tools.register(EditFileTool(tmp_path))
    model = MalformedThenValidAnswerModel()

    answer = run("Fix the bug in a.py", model, tools)

    assert answer == "Fixed the function."
    assert "docstring" not in answer  # the broken JSON must never leak through as the answer
    assert len(model.calls) == 2
    nudge_message = model.calls[1][-1]
    assert nudge_message.role == "user"
    assert "didn't resolve" in nudge_message.content


class SummarizesAsPlainJsonModel(ModelInterface):
    """Reproduces a real live regression: asked to summarize a file, the
    model answers by re-emitting its contents as a plain, syntactically
    valid JSON object -- not a tool call, just an oddly-formatted but
    genuine answer with no tool-call intent at all."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        tool_messages = [m for m in messages if m.role == "tool"]
        if len(tool_messages) == 0:
            return ModelResponse(text=_call_text("read_file", {"path": "config.yaml"}))
        return ModelResponse(
            text='{"endpoint_url": "http://localhost:11434", "model_name": "x"}'
        )


def test_plain_json_answer_with_no_tool_call_intent_is_accepted(tmp_path):
    (tmp_path / "config.yaml").write_text("model:\n  name: x\n", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = SummarizesAsPlainJsonModel()

    answer = run("Summarize config.yaml", model, tools)

    assert answer == '{"endpoint_url": "http://localhost:11434", "model_name": "x"}'
    assert len(model.calls) == 2  # must not burn the whole step budget nudging forever


class WrongShapedArgumentsThenValidAnswerModel(ModelInterface):
    """Reproduces a real gap found in a post-session audit: valid JSON
    naming a real tool, but "arguments" is a bare string instead of an
    object. extract_tool_calls() correctly rejects this (wrong shape),
    and it must still be recognized as an attempted-but-broken call --
    not silently returned as the final answer -- then corrected after
    being nudged."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return ModelResponse(text='{"name": "read_file", "arguments": "config.yaml"}')
        return ModelResponse(text="Fixed the function.")


def test_wrong_shaped_arguments_are_not_returned_as_final_answer(tmp_path):
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = WrongShapedArgumentsThenValidAnswerModel()

    answer = run("Read config.yaml", model, tools)

    assert answer == "Fixed the function."
    assert len(model.calls) == 2
    nudge_message = model.calls[1][-1]
    assert nudge_message.role == "user"
    assert "didn't resolve" in nudge_message.content


class CoincidentalToolNameMentionAtBudgetExhaustionModel(ModelInterface):
    """Reproduces a real gap found in a post-session audit: a model that
    burns the whole step budget on a repeated call, then gives a genuine
    final answer (at the no-tools-offered fallback call) that happens to
    mention a real tool's name in JSON-ish shape with no arguments key --
    this must not be discarded in favor of the generic "could not
    complete" message, since it's an incidental mention, not evidence the
    model is still trying to invoke a tool."""

    def __init__(self):
        self.calls: list[list[Message]] = []

    def generate(self, messages, tools=None):
        self.calls.append(list(messages))
        if tools:
            return ModelResponse(text=_call_text("read_file", {"path": "sample.txt"}))
        return ModelResponse(
            text='{"name": "read_file", "note": "the config mentions read_file settings"}'
        )


def test_coincidental_tool_name_mention_not_discarded_at_final_fallback(tmp_path):
    (tmp_path / "sample.txt").write_text("sample contents", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(tmp_path))
    model = CoincidentalToolNameMentionAtBudgetExhaustionModel()

    answer = run("Summarize sample.txt", model, tools)

    assert answer == '{"name": "read_file", "note": "the config mentions read_file settings"}'
