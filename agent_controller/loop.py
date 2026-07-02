"""Minimal agent loop for the vertical-slice milestone.

Ask the model whether it needs a tool, run at most one tool call if so,
feed the result back, and return a final answer. This proves the model /
tool / controller pieces connect end-to-end — it is intentionally not the
full understand -> plan -> act -> observe -> continue ReAct system yet.
"""
from pathlib import Path

from model_interface.base import ModelInterface
from tools.registry import ToolRegistry

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.md"


def _load_system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def run(user_request: str, model: ModelInterface, tools: ToolRegistry) -> str:
    system_prompt = _load_system_prompt()
    prompt = f"{system_prompt}\n\nUser request: {user_request}"

    response = model.generate(prompt, tools=tools.schemas())

    if not response.tool_calls:
        return response.text

    call = response.tool_calls[0]
    try:
        observation = tools.execute(call.name, call.arguments)
    except Exception as exc:
        observation = f"Error running tool '{call.name}': {exc}"

    follow_up = (
        f"{prompt}\n\n"
        f"Tool call: {call.name}({call.arguments})\n"
        f"Observation:\n{observation}\n\n"
        "Using the observation above, give the user a final answer."
    )
    final_response = model.generate(follow_up, tools=tools.schemas())
    return final_response.text
