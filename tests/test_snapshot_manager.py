import pytest

from state.snapshot import SnapshotManager


def test_snapshot_and_rollback_existing_file(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("original", encoding="utf-8")
    manager = SnapshotManager(tmp_path / ".agent_state")

    snapshot_id = manager.snapshot_before_edit(target)
    target.write_text("modified", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "modified"

    manager.rollback(snapshot_id)
    assert target.read_text(encoding="utf-8") == "original"


def test_snapshot_and_rollback_new_file(tmp_path):
    target = tmp_path / "new_file.txt"
    manager = SnapshotManager(tmp_path / ".agent_state")

    snapshot_id = manager.snapshot_before_edit(target)
    target.write_text("brand new content", encoding="utf-8")
    assert target.is_file()

    manager.rollback(snapshot_id)
    assert not target.exists()


def test_rollback_unknown_id_raises(tmp_path):
    manager = SnapshotManager(tmp_path / ".agent_state")
    with pytest.raises(KeyError):
        manager.rollback("does-not-exist")


def test_history_reflects_recorded_snapshots(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("v1", encoding="utf-8")
    manager = SnapshotManager(tmp_path / ".agent_state")

    snapshot_id = manager.snapshot_before_edit(target)

    history = manager.history()
    assert len(history) == 1
    assert history[0].id == snapshot_id
    assert history[0].existed_before is True
    assert history[0].path == str(target)


def test_manager_persists_across_instances(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("v1", encoding="utf-8")
    state_dir = tmp_path / ".agent_state"

    manager1 = SnapshotManager(state_dir)
    snapshot_id = manager1.snapshot_before_edit(target)
    target.write_text("v2", encoding="utf-8")

    # simulate a new process picking up the same state dir
    manager2 = SnapshotManager(state_dir)
    manager2.rollback(snapshot_id)

    assert target.read_text(encoding="utf-8") == "v1"
