"""Entry point for the local AI coding agent.

Usage:
    python main.py "Summarize config.yaml"
    python main.py            # prompts for a request interactively
"""
import sys

from agent_controller.loop import run
from config import load_config
from model_interface.ollama_adapter import OllamaAdapter
from tools.list_directory import ListDirectoryTool
from tools.read_file import ReadFileTool
from tools.registry import ToolRegistry
from tools.run_command import RunCommandTool
from tools.search_code import SearchCodeTool


def build_tool_registry(project_root: str, command_timeout_seconds: float) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool(project_root))
    registry.register(ListDirectoryTool(project_root))
    registry.register(SearchCodeTool(project_root))
    registry.register(RunCommandTool(project_root, timeout_seconds=command_timeout_seconds))
    return registry


def main() -> None:
    config = load_config()

    model = OllamaAdapter(
        endpoint_url=config["model"]["endpoint_url"],
        model_name=config["model"]["name"],
        request_timeout_seconds=config["model"]["request_timeout_seconds"],
    )
    tools = build_tool_registry(
        config["project"]["root_path"],
        config["execution"]["command_timeout_seconds"],
    )

    user_request = " ".join(sys.argv[1:]) or input("Request: ")
    answer = run(user_request, model, tools)
    print(answer)


if __name__ == "__main__":
    main()
