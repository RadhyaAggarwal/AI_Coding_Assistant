"""Human-in-the-loop confirmation for tools with real side effects.

Path-validated file tools can be *contained* (they simply can't touch
anything outside the project root). A shell command can't be contained
that way — it can do anything a human running it could do, regardless of
its working directory. Tool.requires_confirmation opts a tool into this
gate; read-only tools skip it entirely. The confirm function is
injectable so tests (and, later, a non-CLI frontend) don't have to go
through stdin.

Return value is `bool | str`, not just `bool`: True approves, False
declines with no reason (the original behavior, unchanged), and a
non-empty string declines *with that text as feedback*. Motivated by a
live-observed gap: a decline only ever produced a generic "user
declined" message, giving the model nothing to act on -- unlike a
tools/syntax_check.py rejection, which includes the real parse error.
When a human actually explains why (as happened live: "there would have
been duplicate lines of code"), that reason had nowhere to go before
this. It now flows straight into ToolCallDeniedError's message, which
already reaches the model through agent_controller/loop.py's existing
per-call error-feedback path -- no changes needed there. Every existing
caller using plain True/False is unaffected; this is purely additive.
"""
from typing import Callable

ConfirmFn = Callable[[str], "bool | str"]


class ToolCallDeniedError(Exception):
    """Raised when the human declines to run a confirmation-gated tool call."""


def prompt_confirm(description: str) -> "bool | str":
    raw = input(f"{description}\nAllow? [y/N] (or type a reason to decline with feedback) ").strip()
    lowered = raw.lower()
    if lowered in ("y", "yes"):
        return True
    if not raw or lowered in ("n", "no"):
        return False
    return raw
