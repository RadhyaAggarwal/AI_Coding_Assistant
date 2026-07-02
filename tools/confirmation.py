"""Human-in-the-loop confirmation for tools with real side effects.

Path-validated file tools can be *contained* (they simply can't touch
anything outside the project root). A shell command can't be contained
that way — it can do anything a human running it could do, regardless of
its working directory. Tool.requires_confirmation opts a tool into this
gate; read-only tools skip it entirely. The confirm function is
injectable so tests (and, later, a non-CLI frontend) don't have to go
through stdin.
"""
from typing import Callable

ConfirmFn = Callable[[str], bool]


class ToolCallDeniedError(Exception):
    """Raised when the human declines to run a confirmation-gated tool call."""


def prompt_confirm(description: str) -> bool:
    answer = input(f"{description}\nAllow? [y/N] ").strip().lower()
    return answer in ("y", "yes")
