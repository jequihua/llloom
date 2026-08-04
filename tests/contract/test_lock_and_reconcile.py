"""Contract tests for the workspace lock and journal-backed reconcile."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from llloom.ops import reconcile
from llloom.ops._context import operation
from llloom.state.journal import OperationJournal
from llloom.state.lock import LockError, WorkspaceLock
from llloom.workspace.layout import Workspace


def test_reconcile_noop_when_clean(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    result = reconcile(ws)
    assert result.lock_cleared is False
    assert not result.journals_marked_interrupted


def test_live_lock_refuses_second_operation(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    lock.acquire(op_id="op.holder", owner_id="test")
    with pytest.raises(LockError):
        with operation(ws, op_kind="ingest"):
            raise AssertionError("should not enter")
    lock.release()


def test_stale_lock_and_interrupted_journal_reconcile(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    # Simulate a stale lock owned by an in_progress journal entry.
    lock = WorkspaceLock(ws)
    journal = OperationJournal(ws)
    op_id = journal.new_op_id("ingest")
    acquired = lock.acquire(op_id=op_id, owner_id="test", timeout_seconds=1)
    # Backdate heartbeat.
    past = (datetime.now(timezone.utc) - timedelta(seconds=3600)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    payload = yaml.safe_load(lock.path.read_text(encoding="utf-8"))
    payload["heartbeat_at"] = past
    lock.path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    # Open in_progress journal entry.
    entry = journal.start(op_id=op_id, op_kind="ingest", lock_id=acquired.lock_id)
    assert entry.status == "in_progress"

    # Introduce a stale temp file that reconcile should remove.
    tmp = ws.claims_entities / "temp.yaml.tmp"
    tmp.write_text("partial", encoding="utf-8")

    result = reconcile(ws)
    assert result.lock_cleared is True
    assert op_id in result.journals_marked_interrupted
    assert any(path.endswith("temp.yaml.tmp") for path in result.temp_files_removed)
    # Journal entry should now be interrupted.
    reloaded = journal.load(op_id)
    assert reloaded.status == "interrupted"
    # Lock file should be gone.
    assert not lock.path.is_file()

