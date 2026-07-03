"""Detects whether a candidate final answer actually addresses every
part of the request, using a short, targeted call to the model itself
rather than keyword matching for the actual judgment.

Two prior keyword-based attempts at the judgment itself each caught one
real failure but missed another observed live:
  - "was the confidently-identified expected tool actually called"
    missed a refusal that mentioned the right words ("RepoIndex",
    "class") without giving a real answer, and also depends on lexical
    overlap between the request's wording and a tool's own name — which
    doesn't exist for e.g. "the RepoIndex class" vs. the tool name
    "find_symbol" (no shared vocabulary; a class isn't lexically a
    "symbol").
  - "does the answer text merely mention the topic" is exactly the
    superficial-keyword-match trap this project's tools have hit before
    (find_symbol's fabrication history) — recognizing a refusal or
    non-answer requires actual comprehension, not word overlap.

Both are semantic judgments no amount of regex tuning fixes, so the
actual completeness judgment is now a model call. But a cheap, non-LLM
gate (_looks_compound) still decides whether to bother calling the model
at all — most requests are a single part with nothing to miss, and this
is a one-time-per-request check (unlike tool_router.py, which narrows
tools on every step and would double per-step latency if made
LLM-based), but there's no reason to pay even that bounded cost for a
request that obviously isn't compound.
"""
import re

from agent_controller.text_similarity import tokenize
from model_interface.base import Message, ModelInterface

_PART_SPLIT_PATTERN = re.compile(
    r"\?+|,?\s+and\s+(?:also\s+|additionally\s+)?|\s+also\s+|\s+as well as\s+",
    re.IGNORECASE,
)
_MIN_PART_WORDS = 3

_VERIFY_PROMPT_TEMPLATE = """I asked: "{request}"

Here is a candidate final answer:
"{answer}"

Does this answer genuinely address every distinct part of my request, \
with real information — not a refusal, a guess, or a request for me to \
look it up myself? Reply with exactly the single word COMPLETE if yes. \
Otherwise, reply with one short sentence describing what's missing or \
unanswered. Do not say anything else."""

_COMPLETE_MARKER = "complete"
_MAX_COMPLETE_REPLY_LEN = 40  # a genuine "yes" reply is short; a long one describing a gap is not


def _looks_compound(request: str) -> bool:
    """Cheap, non-LLM pre-check: does this request look like it has more
    than one distinct part (split on '?' and coordinating conjunctions)?
    Used only to decide whether the real completeness check below is
    worth its cost — a single-part request has nothing to miss.
    """
    parts = [p.strip() for p in _PART_SPLIT_PATTERN.split(request) if p.strip()]
    substantial = [p for p in parts if len(tokenize(p)) >= _MIN_PART_WORDS]
    return len(substantial) > 1


def find_unaddressed_part(request: str, answer_text: str, model: ModelInterface) -> str | None:
    """Ask the model whether answer_text actually covers request (only
    if request looks compound to begin with). Returns None if it
    considers the answer complete, or its own short description of
    what's missing otherwise.
    """
    if not _looks_compound(request):
        return None

    prompt = _VERIFY_PROMPT_TEMPLATE.format(request=request, answer=answer_text)
    response = model.generate([Message(role="user", content=prompt)])
    reply = response.text.strip()

    if len(reply) <= _MAX_COMPLETE_REPLY_LEN and _COMPLETE_MARKER in reply.lower():
        return None
    return reply
