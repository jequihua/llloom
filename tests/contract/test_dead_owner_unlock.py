"""Contract tests for Slice 086 guarded local dead-owner unlock.

Pins the load-bearing behavior of the new
``llloom unlock --dead-owner --reason "..."`` mode:

- it clears the workspace lock only when **every** local safety
  predicate passes (same-host, dead PID, matching in_progress
  journal, identical pre-clear re-read, non-empty reason);
- it refuses live, alive-PID, unknown-PID, remote-host,
  missing-metadata, missing-journal, completed-journal, and
  timed-out locks (the last redirects to ``--clear-stale``);
- it is mutually exclusive with ``--clear-stale``;
- the audit journal records ``op_kind="unlock_clear_dead_owner"``
  with the prior op id, owner identity, owner_pid, owner_hostname,
  acquired_at, heartbeat_at, timeout_seconds, ``owner_pid_state=dead``,
  ``race_recheck=passed``, and the operator's reason;
- the frozen stale-recovery rule, ``reconcile``, and
  ``unlock --clear-stale`` are byte-identical;
- the doctor's ``lock:owner-process-dead:<op_id>`` warning
  recommends ``--dead-owner`` only when the journal predicate is
  already satisfied; otherwise it points at wait / reconcile /
  ``--clear-stale``.

Tests monkeypatch ``llloom.state.lock.os.kill`` to drive the
local PID-state probe deterministically; no real processes are
spawned or killed.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from llloom.cli import _build_parser, main as cli_main
from llloom.ops import doctor as run_doctor
from llloom.ops import unlock as run_unlock
from llloom.state.journal import JournalEntry, OperationJournal
from llloom.state.lock import (
    LOCK_FILENAME,
    PID_STATE_ALIVE,
    PID_STATE_DEAD,
    WorkspaceLock,
)
from llloom.workspace.layout import Workspace


# Same improbable-PID convention used by `test_lock_owner_metadata.py`.
_IMPROBABLE_PID = 2_147_483_640


def _force_dead_for_improbable_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the POSIX branch and monkeypatch ``os.kill`` so the
    improbable PID always reads dead.

    Real-process signaling differs across POSIX / Windows; the helper's
    POSIX fallback is deterministic only under a monkeypatched
    ``os.kill``.
    """
    import llloom.state.lock as lock_module

    real_kill = lock_module.os.kill

    def fake_kill(pid: int, sig: int) -> None:
        if pid == _IMPROBABLE_PID:
            raise ProcessLookupError(f"no such process: {pid}")
        return real_kill(pid, sig)

    monkeypatch.setattr(lock_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(lock_module.os, "kill", fake_kill)


def _force_alive_for_improbable_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the POSIX branch and monkeypatch ``os.kill`` so the
    improbable PID always reads alive."""
    import llloom.state.lock as lock_module

    real_kill = lock_module.os.kill

    def fake_kill(pid: int, sig: int) -> None:
        if pid == _IMPROBABLE_PID:
            return None
        return real_kill(pid, sig)

    monkeypatch.setattr(lock_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(lock_module.os, "kill", fake_kill)


def _seed_lock_and_in_progress_journal(
    workspace: Workspace,
    *,
    op_id: str = "op.ingest.dead",
    owner_pid: int = _IMPROBABLE_PID,
    owner_hostname: str | None = None,
    timeout_seconds: int = 300,
    heartbeat_seconds_ago: int = 0,
    omit_owner_fields: bool = False,
    journal_status: str = "in_progress",
    journal_completed_at: str | None = None,
    skip_journal: bool = False,
) -> tuple[Path, str]:
    """Write a hand-crafted lock + matching journal entry to drive the
    dead-owner predicate deterministically. Returns
    ``(lock_path, op_id)``.
    """
    if owner_hostname is None:
        owner_hostname = socket.gethostname()
    now = datetime.now(timezone.utc)
    heartbeat = (
        now - timedelta(seconds=heartbeat_seconds_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload: dict[str, object] = {
        "lock_id": "lock.workspace",
        "scope": "workspace",
        "op_id": op_id,
        "owner_id": f"local.{owner_hostname}.{op_id}",
        "acquired_at": heartbeat,
        "heartbeat_at": heartbeat,
        "timeout_seconds": timeout_seconds,
    }
    if not omit_owner_fields:
        payload["owner_pid"] = owner_pid
        payload["owner_hostname"] = owner_hostname
        payload["owner_cwd"] = str(workspace.root)
        payload["owner_command"] = f"pytest dead-owner fixture {op_id}"
    workspace.state_locks.mkdir(parents=True, exist_ok=True)
    lock_path = workspace.state_locks / LOCK_FILENAME
    lock_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    if not skip_journal:
        journal = OperationJournal(workspace)
        entry = journal.start(
            op_id=op_id, op_kind="ingest", lock_id="lock.workspace"
        )
        if journal_status != "in_progress":
            entry.status = journal_status
        if journal_completed_at is not None:
            entry.completed_at = journal_completed_at
        journal.save(entry)
    return lock_path, op_id


# ---------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------


def test_dead_owner_clears_lock_and_marks_prior_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    lock_path, op_id = _seed_lock_and_in_progress_journal(ws)
    journal = OperationJournal(ws)

    record = run_unlock(
        ws, reason="dev box owner crashed", dead_owner=True
    )

    assert record.refused is False
    assert record.lock_cleared is True
    assert record.mode == "clear_dead_owner_lock"
    assert record.op_id.startswith("op.unlock_clear_dead_owner.")
    assert record.prior_op_id == op_id
    assert record.prior_owner_pid == _IMPROBABLE_PID
    assert record.prior_owner_hostname == socket.gethostname()
    assert record.prior_owner_pid_state == PID_STATE_DEAD
    assert record.target == "workspace"

    # Lock file is gone.
    assert not lock_path.is_file()
    # Prior journal entry is marked interrupted with a note pointing
    # at the audit op id and reason.
    prior_entry = journal.load(op_id)
    assert prior_entry.status == "interrupted"
    assert any(
        record.op_id in note and "dead-owner" in note
        for note in prior_entry.notes
    )
    assert any("dev box owner crashed" in note for note in prior_entry.notes)
    # Audit entry is completed with the right op_kind + notes.
    audit_entry = journal.load(record.op_id)
    assert audit_entry.op_kind == "unlock_clear_dead_owner"
    assert audit_entry.status == "completed"
    assert audit_entry.planned_writes == ["workspace"]
    assert "workspace" in audit_entry.touched_files
    notes_blob = " ".join(audit_entry.notes)
    assert "dead_owner_lock" in notes_blob
    assert "dev box owner crashed" in notes_blob
    assert f"prior_op_id={op_id}" in notes_blob
    assert f"prior_owner_pid={_IMPROBABLE_PID}" in notes_blob
    assert f"prior_owner_hostname={socket.gethostname()}" in notes_blob
    assert "owner_pid_state=dead" in notes_blob
    assert "race_recheck=passed" in notes_blob
    assert "lock_cleared=true" in notes_blob


# ---------------------------------------------------------------------
# 2-7. Refusals
# ---------------------------------------------------------------------


def test_dead_owner_refuses_when_pid_is_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_alive_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    lock_path, op_id = _seed_lock_and_in_progress_journal(ws)

    record = run_unlock(ws, reason="alive test", dead_owner=True)
    assert record.refused is True
    assert record.lock_cleared is False
    assert record.refusal_reason is not None
    assert "alive" in record.refusal_reason.lower()
    assert record.prior_owner_pid_state == PID_STATE_ALIVE
    # Lock file untouched.
    assert lock_path.is_file()
    # Prior journal still in_progress.
    journal = OperationJournal(ws)
    assert journal.load(op_id).status == "in_progress"


def test_dead_owner_refuses_when_metadata_is_missing(
    tmp_path: Path,
) -> None:
    """Old-shape lock with no owner_pid/owner_hostname → refuse."""
    ws = Workspace.init(tmp_path)
    lock_path, op_id = _seed_lock_and_in_progress_journal(
        ws, omit_owner_fields=True
    )

    record = run_unlock(ws, reason="missing meta", dead_owner=True)
    assert record.refused is True
    assert record.lock_cleared is False
    assert "owner_pid" in (record.refusal_reason or "").lower()
    assert lock_path.is_file()


def test_dead_owner_refuses_remote_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    lock_path, op_id = _seed_lock_and_in_progress_journal(
        ws, owner_hostname="some-other-host.example.invalid"
    )

    record = run_unlock(ws, reason="remote host test", dead_owner=True)
    assert record.refused is True
    assert record.lock_cleared is False
    assert "host" in (record.refusal_reason or "").lower()
    assert lock_path.is_file()


def test_dead_owner_refuses_when_journal_entry_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    lock_path, op_id = _seed_lock_and_in_progress_journal(
        ws, skip_journal=True
    )

    record = run_unlock(ws, reason="no journal test", dead_owner=True)
    assert record.refused is True
    assert record.lock_cleared is False
    assert "journal" in (record.refusal_reason or "").lower()
    assert lock_path.is_file()


def test_dead_owner_refuses_when_journal_entry_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    lock_path, op_id = _seed_lock_and_in_progress_journal(ws)
    # Complete the journal entry so the dead-owner predicate must
    # refuse.
    journal = OperationJournal(ws)
    entry = journal.load(op_id)
    entry.status = "completed"
    entry.completed_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    journal.save(entry)

    record = run_unlock(ws, reason="completed journal test", dead_owner=True)
    assert record.refused is True
    assert record.lock_cleared is False
    assert "in_progress" in (record.refusal_reason or "")
    assert lock_path.is_file()
    # Journal entry stays completed (we did not retroactively
    # interrupt it).
    assert journal.load(op_id).status == "completed"


def test_dead_owner_refuses_timed_out_lock_and_points_to_clear_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    lock_path, op_id = _seed_lock_and_in_progress_journal(
        ws, timeout_seconds=1, heartbeat_seconds_ago=3600
    )

    record = run_unlock(ws, reason="timed out test", dead_owner=True)
    assert record.refused is True
    assert record.lock_cleared is False
    assert "--clear-stale" in (record.refusal_reason or "")
    assert lock_path.is_file()


def test_unlock_refuses_when_both_clear_stale_and_dead_owner_supplied(
    tmp_path: Path,
) -> None:
    ws = Workspace.init(tmp_path)
    record = run_unlock(
        ws,
        reason="conflict test",
        clear_stale=True,
        dead_owner=True,
    )
    assert record.refused is True
    assert record.lock_cleared is False
    assert (
        "mutually exclusive" in (record.refusal_reason or "").lower()
        or "at most one" in (record.refusal_reason or "").lower()
    )


# ---------------------------------------------------------------------
# 8. Race re-read
# ---------------------------------------------------------------------


def test_dead_owner_refuses_when_lock_identity_changes_pre_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the lock identity changes between validation and the final
    pre-clear re-read, refuse without mutating anything. Simulated by
    monkeypatching ``WorkspaceLock.read`` to return a renewed
    heartbeat on the second call.
    """
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    lock_path, op_id = _seed_lock_and_in_progress_journal(ws)

    from llloom.state.lock import Lock as LockType
    from llloom.state.lock import WorkspaceLock as WL

    original_read = WL.read
    call_counter = {"n": 0}

    def fake_read(self):
        call_counter["n"] += 1
        result = original_read(self)
        if call_counter["n"] == 2 and result is not None:
            # Return a clone with a different heartbeat to simulate a
            # concurrent renewal.
            return LockType(
                lock_id=result.lock_id,
                scope=result.scope,
                op_id=result.op_id,
                owner_id=result.owner_id,
                acquired_at=result.acquired_at,
                heartbeat_at="2099-01-01T00:00:00Z",
                timeout_seconds=result.timeout_seconds,
                owner_pid=result.owner_pid,
                owner_hostname=result.owner_hostname,
                owner_cwd=result.owner_cwd,
                owner_command=result.owner_command,
            )
        return result

    monkeypatch.setattr(WL, "read", fake_read)

    record = run_unlock(ws, reason="race test", dead_owner=True)
    assert record.refused is True
    assert record.lock_cleared is False
    assert "identity" in (record.refusal_reason or "").lower()
    assert lock_path.is_file()
    # Prior journal entry still in_progress.
    journal = OperationJournal(ws)
    assert journal.load(op_id).status == "in_progress"
    # Slice 086a strengthening: no `op.unlock_clear_dead_owner.*`
    # audit journal entry may have been opened by the race-refusal
    # path. The race refusal happens BEFORE the audit-entry creation
    # so the journal directory must contain no dead-owner op id.
    audit_entries = [
        e for e in journal.iter_entries()
        if e.op_kind == "unlock_clear_dead_owner"
    ]
    assert audit_entries == [], (
        "race-refusal must not leak a half-written dead-owner audit "
        f"entry; got: {[e.op_id for e in audit_entries]}"
    )


# ---------------------------------------------------------------------
# Slice 086a: --dead-owner refuses caller-supplied target at both
# library and CLI layers, without touching lock or prior journal.
# ---------------------------------------------------------------------


def test_dead_owner_library_refuses_non_workspace_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slice 086a regression: the dead-owner library entry point must
    refuse a caller-supplied target. The mode operates only on the
    symbolic ``"workspace"`` target, and refusing structurally before
    reading the lock or touching the journal means no audit entry,
    no prior-journal interruption, and no lock deletion can leak on
    this path.
    """
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    lock_path, op_id = _seed_lock_and_in_progress_journal(ws)
    pre_lock_bytes = lock_path.read_bytes()
    journal = OperationJournal(ws)
    pre_prior_bytes = journal.path(op_id).read_bytes()

    record = run_unlock(
        ws,
        target="not-workspace",
        reason="library target refusal test",
        dead_owner=True,
    )
    assert record.refused is True
    assert record.lock_cleared is False
    assert record.mode == "clear_dead_owner_lock"
    assert record.target == "not-workspace"
    refusal = (record.refusal_reason or "").lower()
    assert "target" in refusal
    assert "workspace" in refusal

    # Lock file and prior journal entry are byte-identical.
    assert lock_path.read_bytes() == pre_lock_bytes
    assert journal.path(op_id).read_bytes() == pre_prior_bytes
    assert journal.load(op_id).status == "in_progress"

    # No `op.unlock_clear_dead_owner.*` audit entry was opened.
    audit_entries = [
        e for e in journal.iter_entries()
        if e.op_kind == "unlock_clear_dead_owner"
    ]
    assert audit_entries == [], (
        "library target refusal must not leak a dead-owner audit "
        f"entry; got: {[e.op_id for e in audit_entries]}"
    )


def test_cli_unlock_dead_owner_refuses_positional_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Slice 086a regression: the CLI must refuse a positional
    target on the dead-owner mode before reaching the library. Exit
    code 1, stderr names the target / ``--dead-owner``, no lock
    deletion, no journal mutation, no audit entry.
    """
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    lock_path, op_id = _seed_lock_and_in_progress_journal(ws)
    pre_lock_bytes = lock_path.read_bytes()
    journal = OperationJournal(ws)
    pre_prior_bytes = journal.path(op_id).read_bytes()

    code = cli_main(
        [
            "--root",
            str(tmp_path),
            "unlock",
            "not-workspace",
            "--dead-owner",
            "--reason",
            "cli target refusal test",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "--dead-owner" in err
    assert "target" in err.lower()
    assert "Traceback" not in err

    # Lock file and prior journal entry are byte-identical.
    assert lock_path.read_bytes() == pre_lock_bytes
    assert journal.path(op_id).read_bytes() == pre_prior_bytes
    assert journal.load(op_id).status == "in_progress"

    # No audit entry exists.
    audit_entries = [
        e for e in journal.iter_entries()
        if e.op_kind == "unlock_clear_dead_owner"
    ]
    assert audit_entries == [], (
        "CLI target refusal must not leak a dead-owner audit entry; "
        f"got: {[e.op_id for e in audit_entries]}"
    )


# ---------------------------------------------------------------------
# 9. CLI success smoke
# ---------------------------------------------------------------------


def test_cli_unlock_dead_owner_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    _seed_lock_and_in_progress_journal(ws)

    code = cli_main(
        [
            "--root",
            str(tmp_path),
            "unlock",
            "--dead-owner",
            "--reason",
            "smoke",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["mode"] == "clear_dead_owner_lock"
    assert payload["lock_cleared"] is True
    assert payload["refused"] is False
    assert payload["target"] == "workspace"
    assert payload["op_id"].startswith("op.unlock_clear_dead_owner.")
    assert payload["prior_owner_pid_state"] == "dead"


def test_cli_unlock_dead_owner_and_clear_stale_conflict_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = Workspace.init(tmp_path)
    code = cli_main(
        [
            "--root",
            str(tmp_path),
            "unlock",
            "--dead-owner",
            "--clear-stale",
            "--reason",
            "conflict",
        ]
    )
    assert code != 0


# ---------------------------------------------------------------------
# 10. CLI verb-count guard
# ---------------------------------------------------------------------


def test_unlock_dead_owner_flag_keeps_verb_count_at_26() -> None:
    parser = _build_parser()
    sub = next(
        a for a in parser._actions if a.dest == "command"  # type: ignore[attr-defined]
    )
    registered = set(sub.choices.keys())  # type: ignore[attr-defined]
    # The new --dead-owner flag is on an existing verb; it must NOT
    # add a new top-level command.
    assert "unlock" in registered
    assert len(registered) == 26


# ---------------------------------------------------------------------
# 11. Doctor recommendation conditional behavior
# ---------------------------------------------------------------------


def test_doctor_recommends_dead_owner_only_when_journal_in_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    _seed_lock_and_in_progress_journal(ws)

    result = run_doctor(ws)
    target = next(
        w
        for w in result.warnings
        if w.warning_id.startswith("lock:owner-process-dead:")
    )
    assert target.recommended_command is not None
    assert "--dead-owner" in target.recommended_command


def test_doctor_does_not_recommend_dead_owner_without_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    # Lock without a matching in_progress journal entry.
    _seed_lock_and_in_progress_journal(ws, skip_journal=True)

    result = run_doctor(ws)
    target = next(
        w
        for w in result.warnings
        if w.warning_id.startswith("lock:owner-process-dead:")
    )
    assert target.recommended_command is not None
    assert "--dead-owner" not in target.recommended_command
    # Honest fallback wording instead.
    assert (
        "reconcile" in target.recommended_command
        or "clear-stale" in target.recommended_command
        or "wait" in target.recommended_command.lower()
    )


def test_doctor_does_not_recommend_dead_owner_when_journal_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    _seed_lock_and_in_progress_journal(ws)
    # Now complete the matching journal entry.
    journal = OperationJournal(ws)
    op_id = "op.ingest.dead"
    entry = journal.load(op_id)
    entry.status = "completed"
    entry.completed_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    journal.save(entry)

    result = run_doctor(ws)
    target = next(
        w
        for w in result.warnings
        if w.warning_id.startswith("lock:owner-process-dead:")
    )
    assert "--dead-owner" not in (target.recommended_command or "")


# ---------------------------------------------------------------------
# Sanity: --clear-stale predicate is byte-identical (no widening)
# ---------------------------------------------------------------------


def test_clear_stale_still_refuses_live_dead_pid_without_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slice 086 must not widen --clear-stale. A live lock with a
    same-host confidently-dead PID and no matching journal entry must
    still refuse `unlock --clear-stale`.
    """
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    lock_path, _ = _seed_lock_and_in_progress_journal(ws, skip_journal=True)

    record = run_unlock(ws, reason="should refuse", clear_stale=True)
    assert record.refused is True
    assert record.lock_cleared is False
    assert lock_path.is_file()
