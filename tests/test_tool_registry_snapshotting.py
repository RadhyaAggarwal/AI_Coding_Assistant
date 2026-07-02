from state.snapshot import SnapshotManager
from tools.edit_file import EditFileTool
from tools.registry import ToolRegistry


def test_registry_snapshots_before_mutating_tool_runs(tmp_path):
    (tmp_path / "sample.py").write_text("original\n", encoding="utf-8")
    snapshots = SnapshotManager(tmp_path / ".agent_state")
    registry = ToolRegistry(confirm=lambda description: True, snapshots=snapshots)
    registry.register(EditFileTool(tmp_path))

    registry.execute("edit_file", {"path": "sample.py", "search": "original", "replace": "changed"})

    assert (tmp_path / "sample.py").read_text(encoding="utf-8") == "changed\n"
    [record] = snapshots.history()
    assert record.path == str((tmp_path / "sample.py").resolve())
    assert record.existed_before is True

    snapshots.rollback(record.id)
    assert (tmp_path / "sample.py").read_text(encoding="utf-8") == "original\n"


def test_registry_does_not_snapshot_read_only_tools(tmp_path):
    from tools.read_file import ReadFileTool

    (tmp_path / "sample.py").write_text("hello\n", encoding="utf-8")
    snapshots = SnapshotManager(tmp_path / ".agent_state")
    registry = ToolRegistry(snapshots=snapshots)
    registry.register(ReadFileTool(tmp_path))

    registry.execute("read_file", {"path": "sample.py"})

    assert snapshots.history() == []


def test_registry_without_snapshots_still_runs_mutating_tools(tmp_path):
    (tmp_path / "sample.py").write_text("original\n", encoding="utf-8")
    registry = ToolRegistry(confirm=lambda description: True)  # no snapshots attached
    registry.register(EditFileTool(tmp_path))

    result = registry.execute(
        "edit_file", {"path": "sample.py", "search": "original", "replace": "changed"}
    )

    assert "Edited" in result
    assert (tmp_path / "sample.py").read_text(encoding="utf-8") == "changed\n"
