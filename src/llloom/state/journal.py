"""Operation journal.

Every mutating operation writes one YAML record under
``state/journals/<op_id>.yaml``. See
``04_specification/storage_and_state_model.md`` Â§"Operation journal shape".
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import yaml

from llloom.workspace.layout import Workspace

# Process-local state for the in-process tie-breaker counter inside
# `OperationJournal.new_op_id`. The full id shape is
# `op.<op_kind>.<YYYYMMDDTHHMMSSffffffZ>.<pid>.<NNN>`: a microsecond
# UTC timestamp + the OS process id + a zero-padded counter. PID gives
# cross-process determinism (separate Python processes always pick
# different PIDs), microsecond resolution makes same-microsecond
# collisions between unrelated processes astronomically rare, and the
# counter handles the rare case of two same-process calls landing in
# the same microsecond. The counter resets when (kind, stamp) advances,
# so memory stays bounded at one entry per distinct op_kind.
_OP_ID_LOCK = threading.Lock()
_OP_ID_LAST: dict[str, tuple[str, int]] = {}


@dataclass
class JournalEntry:
    op_id: str
    op_kind: str
    status: str  # "in_progress" | "completed" | "interrupted" | "refused"
    scope: str = "workspace"
    started_at: str = ""
    completed_at: str | None = None
    lock_id: str = ""
    touched_files: list[str] = field(default_factory=list)
    planned_writes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Persisted summaries of LLMInvoke calls made during this op.
    # Each item is the dict produced by InvocationLog.to_mapping(): typed
    # input class names and content hashes only â€” never raw source text.
    invocation_logs: list[dict] = field(default_factory=list)

    def to_mapping(self) -> dict:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_mapping(cls, data: dict) -> "JournalEntry":
        return cls(
            op_id=str(data["op_id"]),
            op_kind=str(data["op_kind"]),
            status=str(data["status"]),
            scope=str(data.get("scope", "workspace")),
            started_at=str(data.get("started_at", "")),
            completed_at=data.get("completed_at"),
            lock_id=str(data.get("lock_id", "")),
            touched_files=list(data.get("touched_files", []) or []),
            planned_writes=list(data.get("planned_writes", []) or []),
            notes=list(data.get("notes", []) or []),
            invocation_logs=list(data.get("invocation_logs", []) or []),
        )


class OperationJournal:
    """File-backed journal under ``state/journals/``."""

    def __init__(self, workspace: Workspace) -> None:
        self._dir = workspace.state_journals

    def path(self, op_id: str) -> Path:
        return self._dir / f"{op_id}.yaml"

    def exists(self, op_id: str) -> bool:
        return self.path(op_id).is_file()

    def load(self, op_id: str) -> JournalEntry:
        path = self.path(op_id)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return JournalEntry.from_mapping(data)

    def save(self, entry: JournalEntry) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self.path(entry.op_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(entry.to_mapping(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        tmp.replace(path)

    def start(
        self,
        *,
        op_id: str,
        op_kind: str,
        lock_id: str,
        planned_writes: list[str] | None = None,
    ) -> JournalEntry:
        entry = JournalEntry(
            op_id=op_id,
            op_kind=op_kind,
            status="in_progress",
            scope="workspace",
            started_at=_iso_now(),
            lock_id=lock_id,
            planned_writes=list(planned_writes or []),
        )
        self.save(entry)
        return entry

    def complete(
        self,
        entry: JournalEntry,
        *,
        touched_files: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> JournalEntry:
        entry.status = "completed"
        entry.completed_at = _iso_now()
        if touched_files is not None:
            entry.touched_files = list(touched_files)
        if notes:
            entry.notes.extend(notes)
        self.save(entry)
        return entry

    def refuse(self, entry: JournalEntry, reason: str) -> JournalEntry:
        entry.status = "refused"
        entry.completed_at = _iso_now()
        entry.notes.append(reason)
        self.save(entry)
        return entry

    def mark_interrupted(self, op_id: str, note: str | None = None) -> JournalEntry:
        entry = self.load(op_id)
        entry.status = "interrupted"
        if note:
            entry.notes.append(note)
        self.save(entry)
        return entry

    def iter_entries(self) -> Iterator[JournalEntry]:
        if not self._dir.is_dir():
            return iter(())
        return (self.load(p.stem) for p in sorted(self._dir.glob("*.yaml")))

    def latest(self) -> JournalEntry | None:
        entries = list(self.iter_entries())
        if not entries:
            return None
        entries.sort(key=lambda e: e.started_at)
        return entries[-1]

    @staticmethod
    def new_op_id(op_kind: str) -> str:
        """Generate a unique op id of the form
        ``op.<op_kind>.<YYYYMMDDTHHMMSSffffffZ>.<pid>.<NNN>``.

        Three independent disambiguators combine to guarantee
        uniqueness both inside one process and across rapid separate
        Python processes (e.g. successive `llloom.exe` CLI calls):

        - the UTC stamp carries microsecond resolution
          (``strftime("%Y%m%dT%H%M%S%fZ")``), so two calls more than a
          microsecond apart land in distinct buckets;
        - the OS process id (``os.getpid()``) discriminates across
          separate Python processes deterministically even when they
          do hit the same microsecond;
        - a zero-padded process-local counter resets when the
          ``(op_kind, stamp)`` pair advances and handles the rare
          case of two same-process calls landing in the same
          microsecond.

        The ``op.<op_kind>.`` prefix is preserved so `startswith`
        checks elsewhere (e.g. ``op.rebuild.health_report.``) continue
        to match. The id remains a filesystem-safe stem for
        ``state/journals/<op_id>.yaml`` on Windows and POSIX.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        pid = os.getpid()
        with _OP_ID_LOCK:
            last_stamp, counter = _OP_ID_LAST.get(op_kind, ("", 0))
            counter = counter + 1 if stamp == last_stamp else 1
            _OP_ID_LAST[op_kind] = (stamp, counter)
        return f"op.{op_kind}.{stamp}.{pid}.{counter:03d}"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

