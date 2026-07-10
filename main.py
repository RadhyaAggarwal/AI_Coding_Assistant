"""Entry point for the local AI coding agent.

Usage:
    python main.py "Summarize config.yaml"
    python main.py                          # prompts for a request interactively
    python main.py --continue "keep going"  # resume the last conversation, fresh step budget
    python main.py --history                # list recorded file-edit snapshots
    python main.py --rollback <snapshot_id>  # undo a specific edit
"""
import sys
import time
from pathlib import Path

from agent_controller.conversation_store import load_conversation, save_conversation
from agent_controller.loop import is_incomplete_answer, run
from config import load_config
from model_interface.base import Message, ModelUnavailableError
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


def _state_dir(config: dict) -> Path:
    project_root = Path(config["project"]["root_path"]).resolve()
    return project_root / config["state"]["snapshot_dir"]


def build_snapshot_manager(config: dict) -> SnapshotManager:
    return SnapshotManager(_state_dir(config))


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
    state_dir = _state_dir(config)

    if len(sys.argv) >= 2 and sys.argv[1] == "--history":
        print_history(snapshots)
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "--rollback":
        rollback(snapshots, sys.argv[2])
        return

    args = sys.argv[1:]
    continue_previous = bool(args) and args[0] == "--continue"
    if continue_previous:
        args = args[1:]

    model = OllamaAdapter(
        endpoint_url=config["model"]["endpoint_url"],
        model_name=config["model"]["name"],
        request_timeout_seconds=config["model"]["request_timeout_seconds"],
        temperature=config["model"].get("temperature"),
        health_check_timeout_seconds=config["model"].get("health_check_timeout_seconds", 5),
    )
    tools = build_tool_registry(
        config["project"]["root_path"],
        config["execution"]["command_timeout_seconds"],
        snapshots,
    )

    user_request = " ".join(args) or input("Request: ")

    transcript: list[Message] = []
    if continue_previous:
        previous = load_conversation(state_dir)
        if previous is None:
            print("No previous conversation to continue -- starting fresh.")
        else:
            transcript = previous

    try:
        answer = run(
            user_request,
            model,
            tools,
            config["model"]["context_window_tokens"],
            transcript=transcript,
        )
    except ModelUnavailableError as exc:
        print(f"Model unavailable: {exc}")
        return

    save_conversation(state_dir, transcript)
    print(answer)
    if is_incomplete_answer(answer):
        print("\n(Run again with --continue to keep going on this.)")


if __name__ == "__main__":
    main()
