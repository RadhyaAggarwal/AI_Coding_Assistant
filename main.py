"""Entry point for the local AI coding agent.

Usage:
    python main.py "Summarize config.yaml"
    python main.py                          # prompts for a request interactively
    python main.py --history                # list recorded file-edit snapshots
    python main.py --rollback <snapshot_id>  # undo a specific edit
"""
import sys
import time
from pathlib import Path

from agent_controller.loop import run
from config import load_config
from model_interface.ollama_adapter import OllamaAdapter
from state.snapshot import SnapshotManager
from tools.create_file import CreateFileTool
from tools.edit_file import EditFileTool
from tools.find_callers import FindCallersTool
from tools.find_importers import FindImportersTool
from tools.find_symbol import FindSymbolTool
from tools.html_overview import HtmlOverviewTool
from tools.list_directory import ListDirectoryTool
from tools.read_file import ReadFileTool
from tools.registry import ToolRegistry
from tools.repo_overview import RepoOverviewTool
from tools.run_command import RunCommandTool
from tools.search_code import SearchCodeTool


def build_snapshot_manager(config: dict) -> SnapshotManager:
    project_root = Path(config["project"]["root_path"]).resolve()
    return SnapshotManager(project_root / config["state"]["snapshot_dir"])


def build_tool_registry(
    project_root: str,
    command_timeout_seconds: float,
    snapshots: SnapshotManager,
) -> ToolRegistry:
    registry = ToolRegistry(snapshots=snapshots)
    registry.register(ReadFileTool(project_root))
    registry.register(ListDirectoryTool(project_root))
    registry.register(SearchCodeTool(project_root))
    registry.register(RunCommandTool(project_root, timeout_seconds=command_timeout_seconds))
    registry.register(EditFileTool(project_root))
    registry.register(CreateFileTool(project_root))
    registry.register(RepoOverviewTool(project_root))
    registry.register(FindSymbolTool(project_root))
    registry.register(HtmlOverviewTool(project_root))
    registry.register(FindImportersTool(project_root))
    registry.register(FindCallersTool(project_root))
    return registry


def print_history(snapshots: SnapshotManager) -> None:
    records = snapshots.history()
    if not records:
        print("No recorded changes yet.")
        return
    for record in records:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.timestamp))
        action = "edited" if record.existed_before else "created"
        print(f"{record.id}  {when}  {action}  {record.path}")


def rollback(snapshots: SnapshotManager, snapshot_id: str) -> None:
    try:
        print(snapshots.rollback(snapshot_id))
    except KeyError as exc:
        print(exc)


def main() -> None:
    config = load_config()
    snapshots = build_snapshot_manager(config)

    if len(sys.argv) >= 2 and sys.argv[1] == "--history":
        print_history(snapshots)
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "--rollback":
        rollback(snapshots, sys.argv[2])
        return

    model = OllamaAdapter(
        endpoint_url=config["model"]["endpoint_url"],
        model_name=config["model"]["name"],
        request_timeout_seconds=config["model"]["request_timeout_seconds"],
    )
    tools = build_tool_registry(
        config["project"]["root_path"],
        config["execution"]["command_timeout_seconds"],
        snapshots,
    )

    user_request = " ".join(sys.argv[1:]) or input("Request: ")
    answer = run(user_request, model, tools, config["model"]["context_window_tokens"])
    print(answer)


if __name__ == "__main__":
    main()
