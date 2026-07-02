"""Snapshot and rollback system. Per CLAUDE.md, state/ "runs before any
edit": every file-mutating tool call gets its target snapshotted here
first (see ToolRegistry.execute() / Tool.target_path()), so a bad edit
can always be undone — even after the agent process has exited, since
this is disk-backed rather than in-memory.
"""
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SnapshotRecord:
    id: str
    path: str  # absolute path, as a string, for the JSON log
    existed_before: bool
    timestamp: float


class SnapshotManager:
    def __init__(self, state_dir: str | Path):
        self._state_dir = Path(state_dir)
        self._content_dir = self._state_dir / "snapshots"
        self._content_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._state_dir / "log.json"

    def _read_log(self) -> list[dict]:
        if not self._log_path.is_file():
            return []
        return json.loads(self._log_path.read_text(encoding="utf-8"))

    def _write_log(self, entries: list[dict]) -> None:
        self._log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def snapshot_before_edit(self, absolute_path: Path) -> str:
        """Record absolute_path's current state before it's modified.
        Returns a snapshot id that can later be passed to rollback().
        """
        snapshot_id = uuid.uuid4().hex[:12]
        existed = absolute_path.is_file()
        if existed:
            (self._content_dir / snapshot_id).write_bytes(absolute_path.read_bytes())

        record = SnapshotRecord(
            id=snapshot_id,
            path=str(absolute_path),
            existed_before=existed,
            timestamp=time.time(),
        )
        entries = self._read_log()
        entries.append(asdict(record))
        self._write_log(entries)
        return snapshot_id

    def rollback(self, snapshot_id: str) -> str:
        """Restore the file to its pre-edit state, or delete it if it
        didn't exist before the snapshotted edit. Returns a human-readable
        description of what happened.
        """
        entries = self._read_log()
        record = next((e for e in entries if e["id"] == snapshot_id), None)
        if record is None:
            raise KeyError(f"No snapshot found with id '{snapshot_id}'")

        target = Path(record["path"])
        if record["existed_before"]:
            content = (self._content_dir / snapshot_id).read_bytes()
            target.write_bytes(content)
            return f"Restored {target} to its state before snapshot {snapshot_id}."

        if target.is_file():
            target.unlink()
        return f"Deleted {target} (it did not exist before snapshot {snapshot_id})."

    def history(self) -> list[SnapshotRecord]:
        return [SnapshotRecord(**entry) for entry in self._read_log()]
