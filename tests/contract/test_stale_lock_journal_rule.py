"""Contract tests for journal-backed stale-lock recovery.

Frozen rule from
``04_specification/storage_and_state_model.md`` Â§"Journal-backed stale rule":

- expired lock + in-progress journal => recoverable
- expired lock + completed journal => NOT recoverable
- expired lock + missing journal => NOT recoverable

Pre-hardening, ``WorkspaceLock.is_stale`` checked only the heartbeat
deadline; ``reconcile`` cleared any timed-out lock regardless of
journal state. This file exercises all three branches.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from llloom.ops import reconcile
from llloom.state.journal import OperationJournal
from llloom.state.lock import WorkspaceLock
from llloom.workspace.layout import Workspace


def _backdate_lock_heartbeat(lock: WorkspaceLock, seconds_ago: int = 3600) -> None:
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = yaml.safe_load(lock.path.read_text(encoding="utf-8"))
    payload["heartbeat_at"] = past
    lock.path.write_text(
        yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
    )


def test_timed_out_lock_with_in_progress_journal_is_recovered(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    journal = OperationJournal(ws)
    op_id = journal.new_op_id("ingest")
    acquired = lock.acquire(op_id=op_id, owner_id="t", timeout_seconds=1)
    _backdate_lock_heartbeat(lock)
    journal.start(op_id=op_id, op_kind="ingest", lock_id=acquired.lock_id)

    result = reconcile(ws)
    assert result.lock_cleared is True
    assert op_id in result.journals_marked_interrupted
    assert not lock.path.is_file()
    assert journal.load(op_id).status == "interrupted"


def test_timed_out_lock_with_completed_journal_is_not_recovered(tmp_path: Path) -> None:
    """Spec rule (2): a completed journal entry means the op finished
    cleanly; the timed-out lock is forensic, not recoverable."""
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    journal = OperationJournal(ws)
    op_id = journal.new_op_id("ingest")
    acquired = lock.acquire(op_id=op_id, owner_id="t", timeout_seconds=1)
    _backdate_lock_heartbeat(lock)
    entry = journal.start(op_id=op_id, op_kind="ingest", lock_id=acquired.lock_id)
    journal.complete(entry, touched_files=[])
    assert journal.load(op_id).status == "completed"

    result = reconcile(ws)
    assert result.lock_cleared is False, (
        "completed-journal lock must NOT be cleared by reconcile"
    )
    assert not result.journals_marked_interrupted
    # Lock still present; reconcile reports the refusal.
    assert lock.path.is_file()
    assert any("not recoverable" in a for a in result.actions)
    # Journal status remains completed.
    assert journal.load(op_id).status == "completed"


def test_timed_out_lock_with_missing_journal_is_not_recovered(tmp_path: Path) -> None:
    """Spec rule (1) and (2) require a journal entry for the op_id; if
    none exists, the timed-out lock cannot be classified as
    recoverable."""
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    op_id = "op.ingest.20990101T000000Z"
    lock.acquire(op_id=op_id, owner_id="t", timeout_seconds=1)
    _backdate_lock_heartbeat(lock)

    result = reconcile(ws)
    assert result.lock_cleared is False, (
        "lock with no matching journal entry must NOT be cleared as a "
        "normal stale operation"
    )
    assert lock.path.is_file()
    assert any("not recoverable" in a for a in result.actions)
    assert any("no journal entry" in a for a in result.actions)


def test_live_lock_still_refuses_reconcile(tmp_path: Path) -> None:
    """Symmetric positive control."""
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    journal = OperationJournal(ws)
    op_id = journal.new_op_id("ingest")
    lock.acquire(op_id=op_id, owner_id="t")  # no heartbeat backdating
    journal.start(op_id=op_id, op_kind="ingest", lock_id="lock.workspace")

    result = reconcile(ws)
    assert result.lock_cleared is False
    assert any("refuse" in a for a in result.actions)

