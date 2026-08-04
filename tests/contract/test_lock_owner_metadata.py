"""Contract tests for Slice 085 lock owner metadata + read-only diagnostics.

Pins the load-bearing properties of the additive ``Lock.owner_*``
fields, the ``local_owner_pid_state(...)`` helper, the new
``StatusResult.lock_owner_*`` fields, and the doctor
``lock:owner-process-dead:<op_id>`` warning.

The frozen stale-recovery rule (``timeout elapsed + matching
in-progress journal evidence``) is unchanged by this slice and
remains pinned by `test_stale_lock_journal_rule.py` and
`test_unlock_clear_stale.py`. Slice 085 adds *diagnostic*
metadata and *read-only* signals on top of that rule; nothing
here grants a force-unlock or timeout-bypass path.
"""

from __future__ import annotations

import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from llloom.cli import main as cli_main
from llloom.ops import doctor as run_doctor
from llloom.ops import status as run_status
from llloom.ops import unlock as run_unlock
from llloom.ops._context import operation
from llloom.state.journal import OperationJournal
from llloom.state.lock import (
    DEFAULT_TIMEOUT_SECONDS,
    LOCK_FILENAME,
    Lock,
    PID_STATE_ALIVE,
    PID_STATE_DEAD,
    PID_STATE_UNKNOWN,
    WorkspaceLock,
    local_owner_pid_state,
)
from llloom.workspace.layout import Workspace


# An astronomically improbable PID. POSIX ``os.kill(pid, 0)`` would
# raise ``ProcessLookupError`` for an absent process; the Windows
# liveness probe is deliberately different (query-only, no signalling),
# so the dead-PID tests below force the POSIX branch and monkeypatch
# ``os.kill`` to fire the canonical ``ProcessLookupError`` for this
# specific PID. This keeps the helper's POSIX classification path under
# test deterministic without spawning or killing real processes; the
# Windows probe has its own direct coverage.
_IMPROBABLE_PID = 2_147_483_640


def _force_dead_for_improbable_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the POSIX branch and monkeypatch ``os.kill`` so the
    improbable PID always reads dead.

    Real-process signaling differs across POSIX / Windows; the helper's
    POSIX fallback is deterministic only under a monkeypatched
    ``os.kill``. The slice prompt explicitly permits monkeypatching for
    deterministic dead-PID tests.
    """
    import llloom.state.lock as lock_module

    real_kill = lock_module.os.kill

    def fake_kill(pid: int, sig: int) -> None:
        if pid == _IMPROBABLE_PID:
            raise ProcessLookupError(f"no such process: {pid}")
        return real_kill(pid, sig)

    monkeypatch.setattr(lock_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(lock_module.os, "kill", fake_kill)


def _backdate_lock_heartbeat(lock: WorkspaceLock, seconds_ago: int = 3600) -> None:
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = yaml.safe_load(lock.path.read_text(encoding="utf-8"))
    payload["heartbeat_at"] = past
    lock.path.write_text(
        yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
    )


def _write_lock_with(
    workspace: Workspace,
    *,
    op_id: str,
    owner_pid: int | None,
    owner_hostname: str | None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    heartbeat_seconds_ago: int = 0,
    omit_owner_fields: bool = False,
) -> Path:
    """Write a hand-crafted workspace lock with explicit owner metadata.

    Used to construct deterministic same-host-dead-PID fixtures and
    old-shape (no-metadata) fixtures without spawning real processes.
    """
    now = datetime.now(timezone.utc)
    heartbeat = (
        now - timedelta(seconds=heartbeat_seconds_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    acquired = heartbeat
    payload: dict[str, object] = {
        "lock_id": "lock.workspace",
        "scope": "workspace",
        "op_id": op_id,
        "owner_id": f"local.{owner_hostname or 'unknown'}.{op_id}",
        "acquired_at": acquired,
        "heartbeat_at": heartbeat,
        "timeout_seconds": timeout_seconds,
    }
    if not omit_owner_fields:
        payload["owner_pid"] = owner_pid
        payload["owner_hostname"] = owner_hostname
        payload["owner_cwd"] = str(workspace.root)
        payload["owner_command"] = f"pytest fixture for {op_id}"
    workspace.state_locks.mkdir(parents=True, exist_ok=True)
    path = workspace.state_locks / LOCK_FILENAME
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# 1. New operation locks record owner metadata
# ---------------------------------------------------------------------


def test_operation_lock_records_owner_metadata(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    with operation(ws, op_kind="ingest") as ctx:
        assert lock.path.is_file()
        current = lock.read()
        assert current is not None
        assert current.op_id == ctx.op_id
        # Owner metadata fields are populated for the current process.
        assert current.owner_pid == os.getpid()
        assert current.owner_hostname == socket.gethostname()
        assert current.owner_cwd is not None
        assert current.owner_cwd  # non-empty
        assert current.owner_command is not None
        assert current.owner_command  # non-empty
        # The command summary is bounded.
        assert len(current.owner_command) <= 240
        # Same-host check on the running process is confidently alive.
        assert local_owner_pid_state(current) == PID_STATE_ALIVE


def test_lock_yaml_carries_owner_fields_with_deterministic_shape(
    tmp_path: Path,
) -> None:
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    with operation(ws, op_kind="ingest"):
        payload = yaml.safe_load(lock.path.read_text(encoding="utf-8"))
    for key in (
        "owner_pid",
        "owner_hostname",
        "owner_cwd",
        "owner_command",
        "lock_id",
        "scope",
        "op_id",
        "owner_id",
        "acquired_at",
        "heartbeat_at",
        "timeout_seconds",
    ):
        assert key in payload, f"lock YAML missing key {key!r}"
    assert payload["owner_pid"] == os.getpid()
    assert payload["owner_hostname"] == socket.gethostname()


# ---------------------------------------------------------------------
# 2. Old lock YAML remains readable
# ---------------------------------------------------------------------


def test_old_lock_yaml_without_owner_fields_parses_cleanly(
    tmp_path: Path,
) -> None:
    ws = Workspace.init(tmp_path)
    _write_lock_with(
        ws,
        op_id="op.ingest.legacy",
        owner_pid=None,
        owner_hostname=None,
        omit_owner_fields=True,
    )
    lock = WorkspaceLock(ws)
    current = lock.read()
    assert current is not None
    assert current.op_id == "op.ingest.legacy"
    # Optional fields are None when the YAML omits them.
    assert current.owner_pid is None
    assert current.owner_hostname is None
    assert current.owner_cwd is None
    assert current.owner_command is None


def test_old_lock_is_not_reported_as_malformed_by_doctor(
    tmp_path: Path,
) -> None:
    ws = Workspace.init(tmp_path)
    # Hand-write an old lock + matching in-progress journal so the
    # stale-recovery branch can still run without exception. Since
    # heartbeat is now and timeout default is 300 s, the lock is
    # NOT timed out — doctor should see no "malformed" warning for
    # the missing optional fields.
    op_id = "op.ingest.legacy"
    _write_lock_with(
        ws,
        op_id=op_id,
        owner_pid=None,
        owner_hostname=None,
        omit_owner_fields=True,
    )
    result = run_doctor(ws)
    malformed = [w for w in result.warnings if w.warning_id == "lock:malformed"]
    assert malformed == [], (
        f"old-shape lock must not trigger lock:malformed; got "
        f"{[w.warning_id for w in result.warnings]}"
    )


# ---------------------------------------------------------------------
# 3. Status reports pid state without changing recoverability
# ---------------------------------------------------------------------


def test_status_reports_alive_for_current_process_lock(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    with operation(ws, op_kind="ingest"):
        result = run_status(ws)
    assert result.lock_held is True
    assert result.lock_owner_pid == os.getpid()
    assert result.lock_owner_hostname == socket.gethostname()
    assert result.lock_owner_pid_state == PID_STATE_ALIVE
    # Recoverability still governed by the timeout + journal rule.
    assert result.lock_is_timed_out is False
    assert result.lock_recoverable is False
    assert result.recommended_lock_action is not None
    assert "clear-stale" not in (result.recommended_lock_action or "")


def test_status_reports_dead_for_same_host_improbable_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    _write_lock_with(
        ws,
        op_id="op.ingest.dead",
        owner_pid=_IMPROBABLE_PID,
        owner_hostname=socket.gethostname(),
    )
    result = run_status(ws)
    assert result.lock_held is True
    assert result.lock_owner_pid == _IMPROBABLE_PID
    assert result.lock_owner_pid_state == PID_STATE_DEAD
    # Crucially: PID state is "dead" BUT the lock is not timed out and
    # not recoverable — recommended action must NOT say to clear it.
    assert result.lock_is_timed_out is False
    assert result.lock_recoverable is False
    assert "clear-stale" not in (result.recommended_lock_action or "")


def test_status_reports_unknown_for_remote_host(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    _write_lock_with(
        ws,
        op_id="op.ingest.remote",
        owner_pid=12345,
        owner_hostname="some-other-host.example.invalid",
    )
    result = run_status(ws)
    assert result.lock_owner_pid_state == PID_STATE_UNKNOWN


def test_status_reports_unknown_for_old_lock_without_metadata(
    tmp_path: Path,
) -> None:
    ws = Workspace.init(tmp_path)
    _write_lock_with(
        ws,
        op_id="op.ingest.legacy",
        owner_pid=None,
        owner_hostname=None,
        omit_owner_fields=True,
    )
    result = run_status(ws)
    assert result.lock_held is True
    assert result.lock_owner_pid is None
    assert result.lock_owner_hostname is None
    assert result.lock_owner_pid_state == PID_STATE_UNKNOWN


def test_status_reports_none_pid_state_when_no_lock_held(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    result = run_status(ws)
    assert result.lock_held is False
    assert result.lock_owner_pid is None
    assert result.lock_owner_hostname is None
    assert result.lock_owner_cwd is None
    assert result.lock_owner_command is None
    assert result.lock_owner_pid_state is None


# ---------------------------------------------------------------------
# 4. Doctor surfaces dead-owner evidence read-only
# ---------------------------------------------------------------------


def test_doctor_emits_owner_process_dead_warning_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_dead_for_improbable_pid(monkeypatch)
    ws = Workspace.init(tmp_path)
    op_id = "op.ingest.dead"
    _write_lock_with(
        ws,
        op_id=op_id,
        owner_pid=_IMPROBABLE_PID,
        owner_hostname=socket.gethostname(),
    )
    # Provide a matching in-progress journal entry so the situation is
    # plausible.
    journal = OperationJournal(ws)
    journal.start(op_id=op_id, op_kind="ingest", lock_id="lock.workspace")

    # Snapshot the workspace before doctor runs.
    pre_lock_bytes = (ws.state_locks / LOCK_FILENAME).read_bytes()
    pre_journal_bytes = journal.path(op_id).read_bytes()
    pre_pages = sorted(
        p.relative_to(ws.root).as_posix() for p in ws.pages.rglob("*.md")
    )
    pre_claims = sorted(
        p.name for p in ws.claims_entities.glob("*.yaml")
    )

    result = run_doctor(ws)
    warning_ids = [w.warning_id for w in result.warnings]
    target = f"lock:owner-process-dead:{op_id}"
    assert target in warning_ids, (
        f"expected {target} in doctor warnings; got {warning_ids}"
    )
    target_warning = next(w for w in result.warnings if w.warning_id == target)
    assert target_warning.severity == "warning"
    assert target_warning.category == "lock"
    assert "ordinary stale-recovery rule" in target_warning.message
    assert target_warning.recommended_command is not None
    # Recommended command must never be a force path.
    assert "--force" not in target_warning.recommended_command
    assert "force-unlock" not in target_warning.recommended_command
    # Slice 086: the doctor now recommends the explicit guarded
    # local escape hatch `unlock --dead-owner` when the same-host
    # dead-owner predicate AND a matching in_progress journal entry
    # are both satisfied. This test seeds both, so the recommended
    # command should name `--dead-owner` (not "wait", which was the
    # pre-Slice-086 wording when no escape hatch existed).
    assert "--dead-owner" in target_warning.recommended_command
    # Evidence includes the PID-state classification.
    assert any(
        "owner_pid_state=dead" in entry for entry in target_warning.evidence
    )

    # Doctor remains strictly read-only.
    assert (ws.state_locks / LOCK_FILENAME).read_bytes() == pre_lock_bytes
    assert journal.path(op_id).read_bytes() == pre_journal_bytes
    assert sorted(
        p.relative_to(ws.root).as_posix() for p in ws.pages.rglob("*.md")
    ) == pre_pages
    assert sorted(
        p.name for p in ws.claims_entities.glob("*.yaml")
    ) == pre_claims
    # No accepted-warnings file is created by doctor.
    accepted_path = ws.state_reports_health / "accepted_warnings.yaml"
    assert not accepted_path.exists()


# ---------------------------------------------------------------------
# 5. Stale-recovery contract is unchanged even with dead-owner PID
# ---------------------------------------------------------------------


def test_unlock_clear_stale_refuses_live_lock_with_dead_pid(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    op_id = "op.ingest.dead"
    _write_lock_with(
        ws,
        op_id=op_id,
        owner_pid=_IMPROBABLE_PID,
        owner_hostname=socket.gethostname(),
    )
    # No matching journal entry, lock not timed out — even with PID
    # confidently dead, the existing predicate must refuse to clear.
    record = run_unlock(ws, target="--clear-stale", reason="dead pid test",
                       clear_stale=True)
    assert record.lock_cleared is False
    assert record.refused is True
    assert record.refusal_reason is not None
    # Lock file is still on disk.
    assert (ws.state_locks / LOCK_FILENAME).is_file()


# ---------------------------------------------------------------------
# 6. Malformed vs old-metadata distinction
# ---------------------------------------------------------------------


def test_lock_missing_required_field_is_malformed(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    ws.state_locks.mkdir(parents=True, exist_ok=True)
    # Required field ``op_id`` omitted — must still raise/refuse.
    payload = {
        "lock_id": "lock.workspace",
        "scope": "workspace",
        "owner_id": "test",
        "acquired_at": "2026-05-25T00:00:00Z",
        "heartbeat_at": "2026-05-25T00:00:00Z",
        "timeout_seconds": 300,
    }
    (ws.state_locks / LOCK_FILENAME).write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )
    lock = WorkspaceLock(ws)
    with pytest.raises(Exception):
        lock.read()


def test_local_owner_pid_state_zero_or_negative_is_unknown() -> None:
    # Unit-style probe of the helper itself.
    base_lock = Lock(
        lock_id="lock.workspace",
        scope="workspace",
        op_id="op.x",
        owner_id="t",
        acquired_at="2026-05-25T00:00:00Z",
        heartbeat_at="2026-05-25T00:00:00Z",
        timeout_seconds=300,
        owner_pid=0,
        owner_hostname=socket.gethostname(),
    )
    assert local_owner_pid_state(base_lock) == PID_STATE_UNKNOWN
    base_lock.owner_pid = -1
    assert local_owner_pid_state(base_lock) == PID_STATE_UNKNOWN


def test_local_owner_pid_state_unknown_when_hostname_mismatches() -> None:
    lock = Lock(
        lock_id="lock.workspace",
        scope="workspace",
        op_id="op.x",
        owner_id="t",
        acquired_at="2026-05-25T00:00:00Z",
        heartbeat_at="2026-05-25T00:00:00Z",
        timeout_seconds=300,
        owner_pid=os.getpid(),
        owner_hostname="some-other-host.example.invalid",
    )
    assert local_owner_pid_state(lock) == PID_STATE_UNKNOWN


def test_local_owner_pid_state_alive_on_permission_error_and_eperm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slice 085a regression: a same-host owner PID whose
    ``os.kill(pid, 0)`` probe raises ``PermissionError`` OR
    ``OSError`` with ``errno.EPERM`` must be classified ``"alive"``,
    not ``"unknown"``. The OS is confirming that the process exists
    but the current user cannot signal it — a missing process would
    raise ``ProcessLookupError`` / ESRCH instead. Pinned because the
    Slice 085 helper docstring read "permission error → unknown",
    which contradicted the accepted runtime.
    """
    import errno

    import llloom.state.lock as lock_module

    # These fakes exercise the POSIX ``os.kill`` classification branch;
    # force it explicitly so the test is deterministic on Windows too.
    monkeypatch.setattr(lock_module, "_IS_WINDOWS", False)

    lock = Lock(
        lock_id="lock.workspace",
        scope="workspace",
        op_id="op.x",
        owner_id="t",
        acquired_at="2026-05-25T00:00:00Z",
        heartbeat_at="2026-05-25T00:00:00Z",
        timeout_seconds=300,
        owner_pid=os.getpid() + 1,
        owner_hostname=socket.gethostname(),
    )

    def fake_kill_permission_error(pid: int, sig: int) -> None:
        raise PermissionError(f"permission denied for pid={pid}")

    monkeypatch.setattr(lock_module.os, "kill", fake_kill_permission_error)
    assert local_owner_pid_state(lock) == PID_STATE_ALIVE

    def fake_kill_oserror_eperm(pid: int, sig: int) -> None:
        raise OSError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(lock_module.os, "kill", fake_kill_oserror_eperm)
    assert local_owner_pid_state(lock) == PID_STATE_ALIVE


# ---------------------------------------------------------------------
# CLI smoke: `llloom status` JSON now includes lock_owner_* keys
# ---------------------------------------------------------------------


def test_cli_status_emits_lock_owner_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Workspace.init(tmp_path)
    code = cli_main(["--root", str(tmp_path), "status"])
    assert code == 0
    out = capsys.readouterr().out
    # When no lock is held the values are JSON null but the keys are
    # always present on the result dataclass.
    for key in (
        "lock_owner_pid",
        "lock_owner_hostname",
        "lock_owner_cwd",
        "lock_owner_command",
        "lock_owner_pid_state",
    ):
        assert f'"{key}"' in out, f"status JSON missing {key!r}"
