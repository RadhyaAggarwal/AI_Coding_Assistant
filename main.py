"""Entry point for the local AI coding agent.

Usage:
    python main.py "Summarize config.yaml"
    python main.py            # prompts for a request interactively
"""
import sys

from agent_controller.loop import run
from config import load_config
from model_interface.ollama_adapter import OllamaAdapter
from tools.read_file import ReadFileTool
from tools.registry import ToolRegistry


def build_tool_registry(project_root: str) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool(project_root))
    return registry


def main() -> None:
    config = load_config()

    model = OllamaAdapter(
        endpoint_url=config["model"]["endpoint_url"],
        model_name=config["model"]["name"],
        request_timeout_seconds=config["model"]["request_timeout_seconds"],
    )
    tools = build_tool_registry(config["project"]["root_path"])

    user_request = " ".join(sys.argv[1:]) or input("Request: ")
    answer = run(user_request, model, tools)
    print(answer)


if __name__ == "__main__":
    main()
