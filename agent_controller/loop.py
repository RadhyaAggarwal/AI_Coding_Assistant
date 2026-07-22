"""Bounded multi-step agent loop: understand -> act -> observe -> continue.

Ask the model whether it needs a tool, run whatever it calls, feed the
result back, and let it decide whether to call another tool or answer.
This is a real (if simple) ReAct loop now — not the full planner from
the Build Plan, but genuinely multi-step, unlike the earlier one-tool
version. The conversation is proper role-separated messages (system,
user, assistant, tool), which is what makes extending to multiple steps
straightforward instead of a rewrite.

Local models are unreliable in ways that matter a lot more once a loop
can run several steps (Build Plan Addendum 2.2: "degrading plan quality
over long ReAct loops"), so this loop defends against seven failure modes
rather than trusting the model to self-regulate:
  - Not using the backend's structured tool-calling field at all, just
    writing the JSON as plain text. _resolve_tool_calls() recovers that.
  - Planning more than one tool call in a single turn (e.g. a JSON array
    covering both parts of a compound request) and having all but the
    first silently discarded. Observed live: the model correctly planned
    find_importers + find_callers together, only the first ran, and the
    model then reported a false "no calls found" for the one that never
    executed instead of realizing it was never checked. _resolve_tool_calls()
    (plural) and the per-call loop below run every call from one turn,
    not just the first, bounded by _MAX_CALLS_PER_TURN.
  - Passing arguments that don't match the tool's schema, or naming a
    tool that fails outright. Fed back as a "tool" turn so the model can
    correct itself, instead of the loop crashing.
  - Calling the exact same tool with the exact same arguments again
    (getting stuck in a loop) instead of using the observation it
    already has. Detected and refused rather than trusted to the
    system prompt's "don't do that" instruction alone -- but only while
    nothing has changed since the first call. A confirmation-gated tool
    (edit_file/create_file/run_command) succeeding may have changed a
    file, so every prior "already known" result is treated as possibly
    stale from that point on, not permanently deduped. Observed live:
    after a successful edit_file fix, the model correctly tried to
    re-run the exact same run_command test command to verify it, and
    was wrongly refused as "already called" every time, never learning
    the fix had worked -- dedup assumed identical arguments always
    means an identical result, true for a read-only lookup against
    unchanged files, false for a command whose output depends on state
    that just changed. Dedup also fires on an identical call that
    *fails*, not just one that succeeds -- a call is recorded as
    already-tried either way. Without this, a call that fails the same
    way every time (e.g. an edit whose result would be invalid syntax)
    was invisible to dedup, since only successes used to get recorded --
    observed live, a model retried one syntactically-broken edit_file
    call across a full 6-step budget, each attempt burning real budget
    because none had ever succeeded and so none were ever remembered.
    This memory also survives across --continue turns (see the
    already_called parameter below), not just within one run() call --
    otherwise it resets to empty on every fresh invocation, and a model
    can re-run the exact same already-failed search several separate
    --continue turns in a row. Observed live: three consecutive
    --continue turns each ran the identical search_code query and got
    the identical unhelpful result, including on a turn where the human
    had just supplied the correct filename directly -- the dedup guard
    that already prevents this within a turn had no memory of the
    previous turn's attempt. Seeding must still respect the
    clear-on-confirmation-gated-success rule above, not just union every
    past call in -- otherwise a legitimate "fix in turn N, re-verify in
    turn N+1" cross-turn cycle would be wrongly blocked the same way the
    single-turn version of this bug already was. run_command is fully
    exempt from this guard regardless (see Tool.dedup_exempt), not just
    protected by the clear-on-success rule above -- unlike edit_file/
    create_file, where an identical repeat after success is never useful
    (the edited text is already consumed, or the file already has that
    exact content), a repeated run_command call can still be wanted: a
    human re-verifying a fix with the exact same test command wants to
    see it actually run again, not be told it's already known to pass.
    Live-observed: exactly that request, refused by this guard because
    the command had already succeeded once and nothing else had changed
    since -- technically correct that nothing *would* differ, but it
    silently overrode an explicit human request instead of ever reaching
    the confirmation prompt that already gates every run_command call
    regardless of past outcome. edit_file/create_file keep no such
    exemption, since neither has this failure mode.
  - Answering a compound request without actually addressing every part
    of it — a model can produce a confident-sounding answer that's
    really a refusal, a guess, or drops a part it already investigated.
    find_unaddressed_part() (agent_controller/request_coverage.py) asks
    the model itself whether its own answer is complete (a short,
    one-time-per-request check — recognizing a non-answer or a semantic
    match with no shared vocabulary, e.g. "the RepoIndex class" needing
    find_symbol, isn't something keyword matching can do reliably), and
    gets one nudge to fix it before finishing.
  - Attempting a tool call that's broken in some way (fails to parse at
    all, e.g. a docstring's unescaped quotes breaking a JSON string
    value; or parses fine but has the wrong shape, e.g. "arguments" as a
    bare string instead of an object) and having the leftover raw,
    broken-looking text silently returned to the user as if it were a
    genuine final answer — worse than no answer.
    mentions_tool_call_attempt() (model_interface/tool_call_parsing.py)
    requires both a name-key match AND an arguments-like key to treat
    something as an attempt, which is what actually distinguishes a real
    (if broken) attempt from ordinary text that merely mentions a tool's
    name or happens to contain unrelated JSON — see that function's
    docstring for two different live cases that broke a version of this
    check requiring only one signal or the other. A detected attempt gets
    fed back as a nudge to fix the JSON instead of being handed to the
    user.
  - A turn where every call turns out to be a duplicate (see above)
    silently ate a step, giving the model no new information to work
    with. Observed live: a request that made a correct, verified fix
    still failed with "could not complete within 6 steps" -- progress
    messages (see tools/base.py's Tool.progress_message()) added to
    trace this showed only 3 real tool calls executed despite 6 steps
    elapsing, implying the rest were rejected repeats. A duplicate-only
    turn no longer consumes the primary _MAX_STEPS budget, so the model
    gets a genuine shot at answering instead of losing budget to its own
    already-rejected repeats -- but it's bounded by its own small
    _MAX_WASTED_STEPS budget, not exempted outright, so a model that's
    truly stuck repeating itself forever still fails safely rather than
    running unbounded.
A hard step budget bounds all of the above combined, so a model that
never converges can't run forever.

A fabrication risk was tried and reverted here, worth recording so it
isn't re-attempted the same way: describing a fix as already applied
when no edit had actually succeeded (observed live -- a request
exhausted its step budget having only investigated, never called
edit_file, and the final answer still said "I proposed a fix by
modifying..."). The first fix injected a fact into the model's own
context before its answer, grounding it with whether a real change had
happened -- this worked for the case it targeted, but then misfired on
an ordinary info-only query that never needed a fix at all: the same
loop-exit path fires whenever wasted-step budget runs out too (e.g.
several redundant duplicate-call attempts in a row), not just when a
fix was genuinely expected, and the injected note's fix-specific
wording derailed an otherwise-correct answer into describing "no
files were changed" instead of answering the actual question that had
already been fully answered by that point. Reverted in favor of a
non-model-facing fix: main.py now prints a mechanically-derived,
always-accurate summary of which files actually changed (or that none
did) alongside every answer, scanning the transcript's real tool
results directly rather than trusting -- or trying to correct -- the
model's own narration. Same "ground with real facts, don't trust
narration" principle this project already relies on for diffs and
command output, just not fed back into the model's own context, where
it risks influencing behavior it was never meant to affect.
"""
import json
from pathlib import Path

from agent_controller.context_budget import cap_observation, trim_to_budget
from agent_controller.request_coverage import find_unaddressed_part
from agent_controller.tool_router import route_tools
from model_interface.base import (
    Message,
    ModelInterface,
    ModelResponse,
    ModelUnavailableError,
    ToolCall,
)
from model_interface.tool_call_parsing import extract_tool_calls, mentions_tool_call_attempt
from tools.registry import ToolRegistry

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.md"
# Conservative default so existing callers (tests) don't need to pass this;
# main.py passes the real value from config.yaml (model.context_window_tokens).
_DEFAULT_CONTEXT_WINDOW_TOKENS = 8192
# 4 was enough for a single tool call; a real edit -> test -> fix -> retest
# cycle needs more room (read, edit, test-fail, edit-fix, test-pass, then a
# final answer attempt all count as steps). Each step is ~50-240s on this
# machine's CPU-only inference, so this is a real latency tradeoff, not a
# free increase — raise further only if a realistic task still runs out.
# Conservative default so existing callers (tests) don't need to pass this;
# main.py passes the real value from config.yaml (agent.max_steps).
_MAX_STEPS = 6
# Observed live: a request that made a correct fix, verified via two real
# tool calls, still exhausted the whole step budget -- the progress
# messages added to trace this showed only 3 unique tool calls ran despite
# 6 steps elapsing, implying the rest were the model re-issuing calls it
# had already made (silently rejected by the already_called dedup guard
# below) instead of answering. A turn where *every* call turns out to be
# a duplicate makes no real progress, so it doesn't consume the primary
# step budget -- but it's capped by its own small budget rather than
# exempted outright, so a model that's genuinely stuck repeating itself
# forever still fails safely instead of running unbounded (the whole
# reason _MAX_STEPS exists in the first place).
_MAX_WASTED_STEPS = 3
# Observed live: a model can plan several tool calls in a single turn (a
# JSON array covering both parts of a compound request) — a real plan,
# not garbage, and worth executing in full within that one step (see
# _resolve_tool_calls). This bounds how many calls from one turn we'll
# trust and run, the same defensive-against-unreliable-output posture as
# the rest of this loop, in case a bad turn ever lists an implausible
# number of calls.
_MAX_CALLS_PER_TURN = 4
# Shared prefix for both give-up messages (step budget exhausted, or
# stuck repeating itself) -- a single source of truth so a caller (see
# is_incomplete_answer() and main.py's --continue hint) can recognize an
# incomplete answer without duplicating the exact wording.
_INCOMPLETE_ANSWER_PREFIX = "Could not complete the request"
# Measured live (agent_controller/tool_router.py's keyword scoring):
# list_directory scores 0 against a vague, filename-blind bug report --
# its name ("list"+"directory") shares no vocabulary with task-oriented
# words like "bug"/"fix"/"receipt", so it gets narrowed out on every step
# by design, not by mistake. That's the right tradeoff most of the time,
# but costly on exactly the step it matters most: the very first one,
# when nothing is known yet and a directory listing is the cheapest
# possible lead (real filenames like "scratch_receipt.py" are a stronger
# signal than any guessed search term). Force-including these two
# "orientation" tools only on that first step -- not every step, which
# would erode the router's narrowing benefit broadly, a fix already
# rejected once for being too broad -- keeps the cost bounded to one step
# per request while still giving the model the option when it's cheapest
# to offer. Whether the model actually reaches for it once offered is a
# separate, unverified question -- routing controls what's offered, not
# what gets chosen.
_ORIENTATION_TOOL_NAMES = frozenset({"list_directory", "repo_overview"})


def is_incomplete_answer(answer: str) -> bool:
    """True if answer is one of run()'s own give-up messages, rather than
    a genuine (if possibly partial) response the model produced. Callers
    can use this to suggest --continue without needing to know run()'s
    exact wording.
    """
    return answer.startswith(_INCOMPLETE_ANSWER_PREFIX)


def _load_system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _resolve_tool_calls(response: ModelResponse, known_tool_names: set[str]) -> list[ToolCall]:
    if response.tool_calls:
        calls = list(response.tool_calls)
    else:
        calls = extract_tool_calls(response.text, known_tool_names)
    return calls[:_MAX_CALLS_PER_TURN]


def _assistant_turn_content(response: ModelResponse, calls: list[ToolCall]) -> str:
    # Preserve exactly what the model said when it said anything; only
    # synthesize a stand-in when native tool_calls left the text empty.
    if response.text:
        return response.text
    return json.dumps([{"name": call.name, "arguments": call.arguments} for call in calls])


def _call_key(call: ToolCall) -> tuple[str, str]:
    return (call.name, json.dumps(call.arguments, sort_keys=True))


def _sync_transcript(transcript: list[Message] | None, messages: list[Message]) -> None:
    if transcript is not None:
        transcript[:] = messages


def _sync_already_called(
    already_called: set[tuple[str, str]] | None, call_history: set[tuple[str, str]]
) -> None:
    if already_called is not None:
        already_called.clear()
        already_called.update(call_history)


def _sync_call_results(
    call_results_out: dict[tuple[str, str], str] | None, call_results: dict[tuple[str, str], str]
) -> None:
    if call_results_out is not None:
        call_results_out.clear()
        call_results_out.update(call_results)


def run(
    user_request: str,
    model: ModelInterface,
    tools: ToolRegistry,
    context_window_tokens: int = _DEFAULT_CONTEXT_WINDOW_TOKENS,
    transcript: list[Message] | None = None,
    already_called: set[tuple[str, str]] | None = None,
    max_steps: int = _MAX_STEPS,
    call_results: dict[tuple[str, str], str] | None = None,
) -> str:
    """Run one request through the agent loop and return its final answer.

    transcript, if given, serves double duty: on the way in, a non-empty
    list is treated as prior conversation history to continue from (a new
    user turn for user_request is appended after it) rather than starting
    a fresh system+user exchange; on the way out, it's mutated in place
    to hold the complete, untrimmed conversation (including the final
    answer), so a caller can persist it and pass it back in on a later
    call to continue where this one left off. Deliberately full-fidelity,
    not summarized -- see agent_controller/conversation_store.py for the
    persistence side of this and why summarizing was rejected here the
    same way it was for context_budget.py.

    already_called works the same way, for the duplicate-call dedup guard
    below: a seed set of (tool_name, arguments_json) pairs already known
    to have been tried, and mutated in place on the way out so a caller
    can persist and replay it into the next --continue call. Deliberately
    NOT reconstructed from transcript by re-parsing message text on every
    call -- that would duplicate this function's own tool-call parsing
    and its clear-on-confirmation-gated-success rule in a second place;
    persisting the exact set this function already maintains keeps there
    being exactly one place that rule lives.

    call_results is already_called's companion: the real result (or
    failure reason) each already-tried call actually produced, keyed the
    same way, so a duplicate rejection can hand the model that real
    content back instead of just telling it "use the observation you
    already have" and trusting it to correctly find and weight an older
    tool message over its own more recent (possibly wrong) narration --
    live-observed not to work across a --continue chain, which is why
    this persists the same way already_called does rather than staying
    scoped to one run() call. Deliberately a separate dict rather than
    reconstructed from transcript, for the same reason already_called
    isn't: this function already has the exact real value at the moment
    a call succeeds or fails, and persisting that avoids a second,
    easy-to-drift-out-of-sync way of deriving it.
    """
    if transcript:
        messages = list(transcript) + [Message(role="user", content=user_request)]
    else:
        messages = [
            Message(role="system", content=_load_system_prompt()),
            Message(role="user", content=user_request),
        ]
    known_tool_names = {schema["function"]["name"] for schema in tools.schemas()}
    always_keep_names = frozenset(tools.confirmation_required_names())
    dedup_exempt_names = frozenset(tools.dedup_exempt_names())
    always_mutates_names = frozenset(tools.always_mutates_names())
    call_history: set[tuple[str, str]] = set(already_called) if already_called else set()
    result_cache: dict[tuple[str, str], str] = dict(call_results) if call_results else {}
    # Scoped to this one run() call only, never persisted -- a dedup_exempt
    # tool (currently only run_command) isn't checked against the general
    # call_history at all, but is still blocked if it exactly repeats the
    # single most-recently-executed call with nothing else having happened
    # in between. Catches a mindless back-to-back repeat (e.g. retrying an
    # identically-failing command) while still allowing a legitimate rerun
    # the moment anything else occurs -- exactly the case the blanket
    # exemption exists for (re-verifying a fix after a real edit).
    last_executed_call: tuple[str, str] | None = None
    coverage_nudge_used = False
    real_steps_used = 0
    wasted_steps_used = 0
    gave_up_on_repetition = False

    def _generate(sendable, **kwargs):
        # A ModelUnavailableError here would otherwise propagate straight
        # out of run() with transcript/already_called never synced --
        # main.py's caller would then have nothing to save, leaving
        # whatever conversation was last written to disk (possibly a
        # completely different, already-finished one) as the stale thing
        # --continue resumes next. Sync what we have -- even just the
        # pending user turn, if this is the very first call -- so the
        # failed attempt itself is what gets persisted and retried.
        try:
            return model.generate(sendable, **kwargs)
        except ModelUnavailableError:
            _sync_transcript(transcript, messages)
            _sync_already_called(already_called, call_history)
            _sync_call_results(call_results, result_cache)
            raise

    while real_steps_used < max_steps and wasted_steps_used < _MAX_WASTED_STEPS:
        sendable = trim_to_budget(messages, context_window_tokens)
        query_text = " ".join(m.content for m in sendable if m.role != "system")
        is_first_step = real_steps_used == 0 and wasted_steps_used == 0
        step_keep_names = always_keep_names | _ORIENTATION_TOOL_NAMES if is_first_step else always_keep_names
        offered_tools = route_tools(query_text, tools.schemas(), always_keep_names=step_keep_names)
        response = _generate(sendable, tools=offered_tools)
        calls = _resolve_tool_calls(response, known_tool_names)

        if not calls:
            real_steps_used += 1
            if mentions_tool_call_attempt(response.text):
                messages.append(Message(role="assistant", content=response.text))
                messages.append(
                    Message(
                        role="user",
                        content=(
                            "That didn't resolve as a valid tool call — check that "
                            "the JSON is syntactically valid (no unescaped quotes "
                            "breaking a string value, e.g. from an embedded "
                            "docstring) and that 'arguments' is an object, not a "
                            "bare string or other value. No tool ran. Fix it and "
                            "try again, or answer in plain text without "
                            "attempting a tool call."
                        ),
                    )
                )
                continue
            if not coverage_nudge_used:
                try:
                    unaddressed = find_unaddressed_part(user_request, response.text, model)
                except ModelUnavailableError:
                    messages.append(Message(role="assistant", content=response.text))
                    _sync_transcript(transcript, messages)
                    _sync_already_called(already_called, call_history)
                    _sync_call_results(call_results, result_cache)
                    raise
                if unaddressed is not None:
                    coverage_nudge_used = True
                    messages.append(Message(role="assistant", content=response.text))
                    messages.append(
                        Message(
                            role="user",
                            content=(
                                f"Your answer isn't complete: {unaddressed} Please "
                                "address that (using a tool if needed) before "
                                "finishing."
                            ),
                        )
                    )
                    continue
            messages.append(Message(role="assistant", content=response.text))
            _sync_transcript(transcript, messages)
            _sync_already_called(already_called, call_history)
            _sync_call_results(call_results, result_cache)
            return response.text

        messages.append(
            Message(role="assistant", content=_assistant_turn_content(response, calls))
        )

        made_progress = False
        for call in calls:
            is_blocked = (
                _call_key(call) in call_history
                if call.name not in dedup_exempt_names
                else _call_key(call) == last_executed_call
            )
            if is_blocked:
                tools.report(
                    f"Skipping duplicate call to '{call.name}' -- already ran "
                    "with these exact arguments."
                )
                # Hand back the real result (or real failure reason) this
                # exact call already produced, instead of just telling the
                # model to "use the observation you already have" -- live-
                # observed that phrasing isn't enough: given an explicit
                # human instruction to re-verify, the model still trusted
                # its own more recent (fabricated) narration over a real,
                # correct tool result sitting earlier in a long --continue
                # chain, because it was never shown that result again, only
                # told it existed. Falls back to the old generic wording if
                # this specific key somehow has no cached result (shouldn't
                # normally happen now that both the success and failure
                # paths below always cache, but stays safe if it does).
                cached = result_cache.get(_call_key(call))
                if cached is not None:
                    content = (
                        "You already called this exact tool with these exact "
                        f"arguments. Here is that real result again:\n{cached}"
                    )
                else:
                    content = (
                        "You already called this exact tool with these exact "
                        "arguments. Use the observation you already have to "
                        "answer, or call a different tool."
                    )
                messages.append(Message(role="tool", content=content))
                continue

            made_progress = True
            last_executed_call = _call_key(call)
            try:
                observation = tools.execute(call.name, call.arguments)
            except Exception as exc:
                # Record this exact call as already-tried even though it
                # failed -- otherwise a call that fails identically every
                # time (e.g. an edit whose result would be invalid syntax,
                # rejected by tools/syntax_check.py) is invisible to the
                # dedup guard above, since that only ever recorded
                # successes. Observed live: a model kept retrying the
                # exact same syntactically-broken edit_file call across 6
                # full steps -- each attempt looked "new" to dedup because
                # none of them had ever succeeded, so every one burned the
                # primary step budget instead of the bounded wasted-turn
                # one. An identical retry now correctly lands in the
                # cheap, capped wasted-turn path instead.
                call_history.add(_call_key(call))
                result_cache[_call_key(call)] = f"Error: {exc}."
                # Deliberately does NOT say anything like "or answer
                # directly if you can't" -- live-observed, that phrasing
                # directly contradicted the system prompt's "don't ask
                # the user to do this manually" instruction, and being
                # the most recent thing in context before the model's
                # next response, it won. This isn't the right place to
                # grant that permission anyway: the step-budget-
                # exhaustion synthesis call below already is, and only
                # once real attempts are genuinely exhausted -- saying it
                # again here, after a single failure with steps and
                # tools still available, undermines that rather than
                # complementing it. The existing dedup-on-failure guard
                # above (already_called records failures, not just
                # successes) independently prevents this from causing an
                # infinite retry loop, so removing the escape hatch here
                # doesn't reopen that older problem.
                messages.append(
                    Message(
                        role="tool",
                        content=(
                            f"Error: {exc}. Other tools are still available -- try a "
                            "different one or a different approach. Do not tell the "
                            "user to make this change manually; you have tools that "
                            "can do it."
                        ),
                    )
                )
                if call.name in always_mutates_names:
                    # Stop processing the rest of this batch -- live-observed
                    # a model plan four calls in one turn (edit views.py, edit
                    # urls.py to reference what the first edit was supposed to
                    # add, edit a template, then run a check), all decided
                    # before any of them had actually run. The first edit
                    # failed, but the second ran anyway, wiring a URL to a
                    # function that was never added -- exactly the broken
                    # state a real check later caught. Any remaining queued
                    # call here was planned on the same blind premise; break
                    # and let the model react to this real failure on a fresh
                    # turn instead of ploughing through calls that may depend
                    # on an assumption this failure just disproved.
                    break
                continue

            if call.name in always_mutates_names:
                # edit_file/create_file just ran successfully -- both
                # only ever return without raising when they've genuinely
                # written a file (see Tool.always_mutates), so project
                # state may have changed underneath every other cached
                # result. Clearing here means every prior "already known"
                # result is treated as possibly stale from this point on,
                # not just the specific file a tool happened to touch --
                # deliberately conservative, since it's cheap (this set is
                # small) and the alternative is silently blocking a
                # legitimate fix-then-reverify cycle (see run_command's
                # dedup_exempt flag, which handles that specific case
                # directly regardless of this clearing). run_command is
                # deliberately NOT in always_mutates_names, even though it
                # also requires confirmation: it can return without
                # raising after having done nothing at all (a bad path, a
                # typo, a syntax error all just produce an error *string*,
                # not an exception) -- observed live, a run_command call
                # that never actually executed anything still cleared
                # every other tool's cached result, letting the model
                # re-burn real step budget re-doing lookups it already had
                # answers to, for no reason -- the exact thing this guard
                # exists to prevent, reintroduced through this one path.
                call_history.clear()
                result_cache.clear()
            call_history.add(_call_key(call))
            capped = cap_observation(observation)
            result_cache[_call_key(call)] = capped
            messages.append(Message(role="tool", content=capped))
            if call.name in always_mutates_names:
                # Same reasoning as the failure-path break above, for the
                # success case: a later call still queued in this same
                # batch was planned before this mutation's real outcome was
                # known at all, batching-blind to it either way -- even a
                # successful edit may not be the one the next queued call
                # assumed (a different file, a different shape). Break so
                # the model reacts to what actually happened on a fresh
                # turn rather than continuing a plan formed in the dark.
                break

        if made_progress:
            real_steps_used += 1
        else:
            # Every call this turn was a duplicate -- no new information
            # reached the model, so this didn't consume the primary
            # budget. Still bounded by _MAX_WASTED_STEPS, so a model
            # that's genuinely stuck repeating itself forever still stops
            # instead of running unbounded.
            wasted_steps_used += 1
            if wasted_steps_used >= _MAX_WASTED_STEPS:
                gave_up_on_repetition = True

    # Step budget exhausted (either the real one, or the smaller one
    # reserved for duplicate-only turns). Ask once more, without tool
    # access, so the model synthesizes an answer from whatever it
    # gathered rather than the loop just giving up with nothing. If it
    # still tries to call a tool anyway (native structured call, or the
    # same name+arguments signature mentions_tool_call_attempt() checks
    # for elsewhere), it hasn't registered that it's out of tools -- give
    # it exactly one more turn with an explicit instruction to stop
    # trying and describe its partial progress in plain language instead
    # (the same trusted mechanism a normal successful answer already
    # uses, just pointed at "explain what you have" instead of "answer
    # the question" -- not a new, less-reliable summarization path).
    # Only if it *still* can't produce real prose after being told
    # directly is this a genuine, unrecoverable give-up.
    final_response = _generate(trim_to_budget(messages, context_window_tokens))
    if final_response.tool_calls or mentions_tool_call_attempt(final_response.text):
        messages.append(Message(role="assistant", content=final_response.text))
        messages.append(
            Message(
                role="user",
                content=(
                    "No more tools are available for this request -- don't "
                    "attempt another tool call, it won't run. In a few "
                    "plain-language sentences, explain what you were able "
                    "to find out so far, how it relates to what was asked, "
                    "and what's still unanswered."
                ),
            )
        )
        final_response = _generate(trim_to_budget(messages, context_window_tokens))
        if final_response.tool_calls or mentions_tool_call_attempt(final_response.text):
            if gave_up_on_repetition:
                final_text = (
                    f"{_INCOMPLETE_ANSWER_PREFIX} -- kept repeating "
                    "already-answered tool calls without making progress."
                )
            else:
                final_text = f"{_INCOMPLETE_ANSWER_PREFIX} within {max_steps} steps."
            messages.append(Message(role="assistant", content=final_text))
            _sync_transcript(transcript, messages)
            _sync_already_called(already_called, call_history)
            _sync_call_results(call_results, result_cache)
            return final_text

    messages.append(Message(role="assistant", content=final_response.text))
    _sync_transcript(transcript, messages)
    _sync_already_called(already_called, call_history)
    _sync_call_results(call_results, result_cache)
    return final_response.text
