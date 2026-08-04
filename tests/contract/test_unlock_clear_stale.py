"""Contract: truthful `unlock` and guarded stale-lock recovery.

Pins the Slice 069 contract from
``feedback/2026-05-22_llloom_development_roadmap_synthesis.md``:

- bare ``unlock <target> --reason ...`` records a maintenance
  window in the journal and **never** clears the workspace lock;
- ``unlock --clear-stale --reason ...`` clears the workspace lock
  only when
  :meth:`llloom.state.lock.WorkspaceLock.is_stale_recoverable`
  returns ``(True, ...)`` (timed-out lock + matching in-progress
  journal entry);
- live, completed-journal, and missing-journal locks all refuse
  without deleting the lock file;
- successful clearing marks the prior journal entry ``interrupted``
  and writes a fresh completed audit journal entry naming the
  prior op id / owner / heartbeat plus the human-supplied reason;
- ``status(...)`` exposes additive lock metadata so an agent can
  see recoverability without inspecting YAML.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from llloom.cli import main as cli_main
from llloom.ops import status, unlock
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


def _seed_stale_recoverable_lock(
    ws: Workspace, op_kind: str = "ingest"
) -> tuple[WorkspaceLock, OperationJournal, str]:
    lock = WorkspaceLock(ws)
    journal = OperationJournal(ws)
    op_id = journal.new_op_id(op_kind)
    acquired = lock.acquire(op_id=op_id, owner_id="prior-owner", timeout_seconds=1)
    _backdate_lock_heartbeat(lock)
    journal.start(op_id=op_id, op_kind=op_kind, lock_id=acquired.lock_id)
    return lock, journal, op_id


def _seed_live_lock(ws: Workspace) -> tuple[WorkspaceLock, OperationJournal, str]:
    lock = WorkspaceLock(ws)
    journal = OperationJournal(ws)
    op_id = journal.new_op_id("ingest")
    acquired = lock.acquire(op_id=op_id, owner_id="prior-owner")
    journal.start(op_id=op_id, op_kind="ingest", lock_id=acquired.lock_id)
    return lock, journal, op_id


def _seed_stale_completed_lock(
    ws: Workspace,
) -> tuple[WorkspaceLock, OperationJournal, str]:
    lock = WorkspaceLock(ws)
    journal = OperationJournal(ws)
    op_id = journal.new_op_id("ingest")
    acquired = lock.acquire(op_id=op_id, owner_id="prior-owner", timeout_seconds=1)
    _backdate_lock_heartbeat(lock)
    entry = journal.start(op_id=op_id, op_kind="ingest", lock_id=acquired.lock_id)
    journal.complete(entry, touched_files=[])
    return lock, journal, op_id


def _seed_stale_missing_journal_lock(
    ws: Workspace,
) -> tuple[WorkspaceLock, str]:
    lock = WorkspaceLock(ws)
    op_id = "op.ingest.20990101T000000000000Z.99999.001"
    lock.acquire(op_id=op_id, owner_id="prior-owner", timeout_seconds=1)
    _backdate_lock_heartbeat(lock)
    return lock, op_id


def test_bare_unlock_does_not_clear_workspace_lock(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    lock, _, prior_op_id = _seed_stale_recoverable_lock(ws)
    assert lock.path.is_file()

    result = unlock(ws, target="workspace", reason="opening maintenance window")

    assert result.mode == "unlock_window"
    assert result.lock_cleared is False
    assert result.refused is False
    assert result.expires_at, "bare unlock window must populate expires_at"
    assert result.op_id.startswith("op.unlock.")
    assert lock.path.is_file(), "bare unlock must NEVER delete the lock file"
    # The prior journal entry must be untouched by the bare unlock.
    prior = OperationJournal(ws).load(prior_op_id)
    assert prior.status == "in_progress"


def test_clear_stale_refuses_when_no_lock(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    assert not lock.path.is_file()

    result = unlock(ws, reason="nothing to clear", clear_stale=True)

    assert result.mode == "clear_stale_lock"
    assert result.refused is True
    assert result.lock_cleared is False
    assert result.refusal_reason
    assert "no workspace lock" in result.refusal_reason.lower()
    assert not lock.path.is_file()


def test_clear_stale_refuses_live_lock(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    lock, journal, prior_op_id = _seed_live_lock(ws)
    assert lock.path.is_file()

    result = unlock(ws, reason="live lock is not stale", clear_stale=True)

    assert result.refused is True
    assert result.lock_cleared is False
    assert result.refusal_reason
    lower = result.refusal_reason.lower()
    assert "live" in lower or "not timed out" in lower
    assert lock.path.is_file()
    # Prior journal entry untouched.
    assert journal.load(prior_op_id).status == "in_progress"


def test_clear_stale_refuses_completed_journal(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    lock, journal, prior_op_id = _seed_stale_completed_lock(ws)
    assert lock.path.is_file()
    assert journal.load(prior_op_id).status == "completed"

    result = unlock(
        ws,
        reason="completed-journal lock is forensic, not recoverable",
        clear_stale=True,
    )

    assert result.refused is True
    assert result.lock_cleared is False
    assert result.refusal_reason
    assert "completed" in result.refusal_reason.lower()
    # Lock still present; prior journal still completed (not interrupted).
    assert lock.path.is_file()
    assert journal.load(prior_op_id).status == "completed"


def test_clear_stale_refuses_missing_journal(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    lock, prior_op_id = _seed_stale_missing_journal_lock(ws)
    assert lock.path.is_file()

    result = unlock(
        ws,
        reason="no matching journal entry for this lock op",
        clear_stale=True,
    )

    assert result.refused is True
    assert result.lock_cleared is False
    assert result.refusal_reason
    lower = result.refusal_reason.lower()
    assert "no journal entry" in lower or "missing" in lower
    assert lock.path.is_file()
    # No journal entry should have been silently fabricated for the prior op.
    assert not OperationJournal(ws).exists(prior_op_id)


def test_clear_stale_succeeds_for_timed_out_in_progress_journal(
    tmp_path: Path,
) -> None:
    ws = Workspace.init(tmp_path)
    lock, journal, prior_op_id = _seed_stale_recoverable_lock(ws)
    assert lock.path.is_file()

    result = unlock(
        ws,
        reason="stale render lock from crashed coder",
        clear_stale=True,
    )

    assert result.mode == "clear_stale_lock"
    assert result.refused is False
    assert result.lock_cleared is True
    assert result.prior_op_id == prior_op_id
    assert result.prior_owner_id == "prior-owner"
    assert result.prior_acquired_at
    assert result.prior_heartbeat_at
    assert result.op_id.startswith("op.unlock_clear_stale.")
    # Lock file removed; prior journal marked interrupted.
    assert not lock.path.is_file()
    assert journal.load(prior_op_id).status == "interrupted"


def test_clear_stale_writes_audit_journal_entry(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    lock, journal, prior_op_id = _seed_stale_recoverable_lock(ws)

    result = unlock(
        ws,
        reason="stale lock from crashed render in M042",
        clear_stale=True,
    )
    assert result.lock_cleared is True

    audit = journal.load(result.op_id)
    assert audit.op_kind == "unlock_clear_stale"
    assert audit.status == "completed"
    assert audit.completed_at
    notes_blob = " ".join(audit.notes)
    assert "clear_stale_lock" in notes_blob
    assert prior_op_id in notes_blob
    assert "prior-owner" in notes_blob
    assert "M042" in notes_blob  # human reason preserved verbatim
    assert "lock_cleared=true" in notes_blob

    # Prior journal entry should mention the clearing op id and reason.
    prior = journal.load(prior_op_id)
    assert prior.status == "interrupted"
    prior_notes = " ".join(prior.notes)
    assert result.op_id in prior_notes
    assert "M042" in prior_notes


def test_cli_clear_stale_requires_reason_and_exits_nonzero_on_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Workspace.init(tmp_path)
    # No reason at all: argparse refuses with exit 2 (CLI usage error).
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["--root", str(tmp_path), "unlock", "--clear-stale"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--reason" in err

    # Reason supplied but no lock present: structured refusal, exit 1.
    rc = cli_main(
        [
            "--root",
            str(tmp_path),
            "unlock",
            "--clear-stale",
            "--reason",
            "no lock to clear",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert '"refused": true' in out
    assert '"lock_cleared": false' in out
    assert "no workspace lock" in out.lower()


def test_status_reports_lock_recoverability_metadata(tmp_path: Path) -> None:
    # Case A: no lock — recoverability fields are None/False.
    ws_clean = Workspace.init(tmp_path / "clean")
    s_clean = status(ws_clean)
    assert s_clean.lock_held is False
    assert s_clean.lock_op_id is None
    assert s_clean.lock_is_timed_out is False
    assert s_clean.lock_recoverable is False
    assert s_clean.recommended_lock_action is None

    # Case B: live lock — recoverable False, action names waiting.
    ws_live = Workspace.init(tmp_path / "live")
    _seed_live_lock(ws_live)
    s_live = status(ws_live)
    assert s_live.lock_held is True
    assert s_live.lock_is_timed_out is False
    assert s_live.lock_recoverable is False
    assert s_live.lock_recoverability_reason == "lock has not timed out"
    assert s_live.recommended_lock_action is not None
    assert "wait" in s_live.recommended_lock_action.lower()

    # Case C: stale recoverable — recoverable True, action names clear-stale.
    ws_rec = Workspace.init(tmp_path / "recoverable")
    _seed_stale_recoverable_lock(ws_rec)
    s_rec = status(ws_rec)
    assert s_rec.lock_held is True
    assert s_rec.lock_is_timed_out is True
    assert s_rec.lock_recoverable is True
    assert s_rec.recommended_lock_action is not None
    assert "--clear-stale" in s_rec.recommended_lock_action

    # Case D: stale unrecoverable (completed journal) — recoverable False,
    # action names manual investigation.
    ws_unrec = Workspace.init(tmp_path / "unrecoverable")
    _seed_stale_completed_lock(ws_unrec)
    s_unrec = status(ws_unrec)
    assert s_unrec.lock_held is True
    assert s_unrec.lock_is_timed_out is True
    assert s_unrec.lock_recoverable is False
    assert s_unrec.lock_recoverability_reason is not None
    assert "completed" in s_unrec.lock_recoverability_reason.lower()
    assert s_unrec.recommended_lock_action is not None
    assert "manual" in s_unrec.recommended_lock_action.lower()
