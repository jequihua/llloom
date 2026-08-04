"""`unlock` operation.

Three modes, all routed through the same CLI/library verb:

- **Bare unlock window** — preserves the legacy behavior. Opens a
  time-bounded unlock window in the operation journal for the
  caller-named ``target``. **Never** touches the workspace lock
  file. Returns ``UnlockRecord(mode="unlock_window",
  lock_cleared=False, ...)``. Use for human/agent maintenance
  windows where the audit record is the point.
- **Guarded stale-lock clear** — ``unlock(..., clear_stale=True,
  reason="...")`` clears the workspace lock file only when the
  frozen
  :meth:`llloom.state.lock.WorkspaceLock.is_stale_recoverable`
  predicate returns ``(True, ...)``: the lock is timed out AND the
  matching journal entry exists AND is ``in_progress``. On success
  the prior journal entry is marked ``interrupted`` with a note
  pointing at the clearing op id and the human-supplied reason; a
  fresh completed audit entry (``op_kind="unlock_clear_stale"``)
  records prior op id / owner / heartbeat / reason / cleared flag.
  Refusal returns a structured ``UnlockRecord`` with
  ``refused=True`` and ``refusal_reason`` set; the workspace lock
  file is never deleted on the refusal path.
- **Guarded dead-owner clear** (Slice 086) —
  ``unlock(..., dead_owner=True, reason="...")`` is a narrow local
  same-host operator escape hatch for "the process that owns this
  workspace lock is gone, the journal says it was still in progress,
  and I am explicitly choosing to clear it with an audit trail."
  Mutually exclusive with ``clear_stale``. Clears the workspace
  lock file only when **every** local safety predicate passes:
  (1) the workspace lock exists and parses cleanly; (2)
  ``owner_pid`` is present and ``> 0``; (3) ``owner_hostname``
  matches ``socket.gethostname()``; (4)
  ``local_owner_pid_state(lock) == PID_STATE_DEAD``; (5) the
  matching journal entry for ``lock.op_id`` exists; (6) that
  journal entry has ``status == "in_progress"``; (7) that journal
  entry has ``completed_at is None``; (8) the operator supplied a
  non-empty reason; (9) a final pre-clear re-read of the lock
  file matches the same identity (`lock_id`, `op_id`, `owner_id`,
  `owner_pid`, `owner_hostname`, `acquired_at`, `heartbeat_at`,
  `timeout_seconds`). Refuses if the lock is already timed out
  (pointing operator at ``--clear-stale`` / ``reconcile``) so the
  two recovery modes stay distinct. On success the prior journal
  entry is marked ``interrupted`` and a fresh completed audit
  entry (``op_kind="unlock_clear_dead_owner"``) is written with
  bounded notes. The frozen
  :meth:`llloom.state.lock.WorkspaceLock.is_stale_recoverable`
  predicate is unchanged; ``reconcile`` continues to use only the
  stale-lock rule.

Slice 069 from
``feedback/2026-05-22_llloom_development_roadmap_synthesis.md``
extended this op without expanding the CLI verb count. Slice 086
adds the dead-owner mode under the same operation, keeping the
top-level verb count at 26. Live locks, completed-journal locks,
and missing-journal locks are still refused on every mode;
``reconcile``'s automatic-repair semantics are unchanged.
"""

from __future__ import annotations

import socket
from datetime import datetime, timedelta, timezone

from llloom.ops._context import iso_now
from llloom.ops.results import UnlockRecord
from llloom.state.journal import OperationJournal
from llloom.state.lock import (
    LockError,
    PID_STATE_DEAD,
    WorkspaceLock,
    local_owner_pid_state,
)
from llloom.workspace.layout import Workspace


DEFAULT_UNLOCK_DURATION_SECONDS = 600
CLEAR_STALE_OP_KIND = "unlock_clear_stale"
CLEAR_DEAD_OWNER_OP_KIND = "unlock_clear_dead_owner"
UNLOCK_WINDOW_OP_KIND = "unlock"


def unlock(
    workspace: Workspace,
    *,
    target: str | None = None,
    reason: str,
    clear_stale: bool = False,
    dead_owner: bool = False,
    duration_seconds: int = DEFAULT_UNLOCK_DURATION_SECONDS,
) -> UnlockRecord:
    if clear_stale and dead_owner:
        return _refused(
            target=target or "workspace",
            reason=reason if isinstance(reason, str) else "",
            mode="clear_dead_owner_lock",
            refusal_reason=(
                "--clear-stale and --dead-owner are mutually exclusive; "
                "pass at most one"
            ),
        )

    if not isinstance(reason, str) or not reason.strip():
        if dead_owner:
            mode = "clear_dead_owner_lock"
        elif clear_stale:
            mode = "clear_stale_lock"
        else:
            mode = "unlock_window"
        return _refused(
            target=target or "",
            reason=reason if isinstance(reason, str) else "",
            mode=mode,
            refusal_reason="reason is required and must be non-empty",
        )

    if dead_owner:
        return _clear_dead_owner(workspace, reason=reason, target=target)
    if clear_stale:
        return _clear_stale(workspace, reason=reason, target=target)
    return _unlock_window(
        workspace,
        target=target,
        reason=reason,
        duration_seconds=duration_seconds,
    )


def _unlock_window(
    workspace: Workspace,
    *,
    target: str | None,
    reason: str,
    duration_seconds: int,
) -> UnlockRecord:
    if not isinstance(target, str) or not target.strip():
        return _refused(
            target=target or "",
            reason=reason,
            mode="unlock_window",
            refusal_reason=(
                "target is required for the bare unlock window mode; "
                "use --clear-stale (no target) to clear a stale lock"
            ),
        )

    journal = OperationJournal(workspace)
    op_id = journal.new_op_id(UNLOCK_WINDOW_OP_KIND)
    entry = journal.start(
        op_id=op_id,
        op_kind=UNLOCK_WINDOW_OP_KIND,
        lock_id="(none)",
        planned_writes=[target],
    )
    entry.notes.append(
        f"unlock_window target={target!r} reason={reason!r} "
        f"lock_cleared=false"
    )
    now = iso_now()
    expires = (
        datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    journal.complete(
        entry,
        touched_files=[target],
        notes=[f"expires_at={expires}"],
    )
    return UnlockRecord(
        target=target,
        reason=reason,
        unlocked_at=now,
        expires_at=expires,
        op_id=op_id,
        mode="unlock_window",
        lock_cleared=False,
    )


def _clear_stale(
    workspace: Workspace,
    *,
    reason: str,
    target: str | None,
) -> UnlockRecord:
    lock = WorkspaceLock(workspace)
    journal = OperationJournal(workspace)
    symbolic_target = target if (isinstance(target, str) and target) else "workspace"

    try:
        current = lock.read()
    except LockError as exc:
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_stale_lock",
            refusal_reason=f"workspace lock file is malformed: {exc}",
        )

    if current is None:
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_stale_lock",
            refusal_reason="no workspace lock present",
        )

    if not lock.is_timed_out(current):
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_stale_lock",
            refusal_reason=(
                f"workspace lock is live (not timed out); "
                f"op_id={current.op_id} owner_id={current.owner_id} "
                f"heartbeat_at={current.heartbeat_at}"
            ),
            prior=current,
        )

    recoverable, recovery_reason = lock.is_stale_recoverable(
        current, journal=journal
    )
    if not recoverable:
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_stale_lock",
            refusal_reason=(
                f"timed-out lock is not stale-recoverable: "
                f"{recovery_reason}"
            ),
            prior=current,
        )

    # Recoverable: open the audit entry, mark prior interrupted,
    # then clear the lock file. Order matters: prior-journal note
    # must reference the new audit op_id, but the audit entry must
    # also remember the prior op id, so we resolve the audit op_id
    # before mutating the prior journal entry.
    op_id = journal.new_op_id(CLEAR_STALE_OP_KIND)
    audit_entry = journal.start(
        op_id=op_id,
        op_kind=CLEAR_STALE_OP_KIND,
        lock_id=current.lock_id,
        planned_writes=[symbolic_target],
    )
    audit_entry.notes.append(
        f"clear_stale_lock reason={reason!r} "
        f"prior_op_id={current.op_id} "
        f"prior_owner_id={current.owner_id} "
        f"prior_acquired_at={current.acquired_at} "
        f"prior_heartbeat_at={current.heartbeat_at} "
        f"prior_timeout_seconds={current.timeout_seconds} "
        f"recoverability={recovery_reason}"
    )
    journal.mark_interrupted(
        current.op_id,
        note=(
            f"interrupted by unlock --clear-stale {op_id} "
            f"reason={reason!r}"
        ),
    )
    lock.clear()
    journal.complete(
        audit_entry,
        touched_files=[symbolic_target],
        notes=["lock_cleared=true"],
    )

    now = iso_now()
    return UnlockRecord(
        target=symbolic_target,
        reason=reason,
        unlocked_at=now,
        expires_at="",
        op_id=op_id,
        mode="clear_stale_lock",
        lock_cleared=True,
        refused=False,
        prior_op_id=current.op_id,
        prior_owner_id=current.owner_id,
        prior_acquired_at=current.acquired_at,
        prior_heartbeat_at=current.heartbeat_at,
    )


def _clear_dead_owner(
    workspace: Workspace,
    *,
    reason: str,
    target: str | None,
) -> UnlockRecord:
    """Slice 086 — guarded local same-host dead-owner clear.

    See the module docstring for the full predicate list. Each
    refusal returns an :class:`UnlockRecord` with ``refused=True``
    and leaves both the workspace lock file and the prior journal
    entry untouched. Slice 086a tightening: the ``target`` arg
    must be omitted or empty — the dead-owner mode operates only
    on the symbolic ``"workspace"`` target, and a caller-supplied
    target refuses structurally before any lock or journal is
    touched.
    """
    # Slice 086a: refuse any non-empty caller-supplied target.
    # The dead-owner mode is workspace-only. Refusing before
    # reading the lock or touching the journal means no audit
    # entry, no prior-journal interruption, and no lock deletion
    # can leak on this path.
    if isinstance(target, str) and target.strip() and target != "workspace":
        return _refused(
            target=target,
            reason=reason,
            mode="clear_dead_owner_lock",
            refusal_reason=(
                f"--dead-owner does not accept a target (got "
                f"{target!r}); the mode operates only on the symbolic "
                "workspace target — omit the positional target"
            ),
        )

    symbolic_target = "workspace"
    lock = WorkspaceLock(workspace)
    journal = OperationJournal(workspace)

    try:
        current = lock.read()
    except LockError as exc:
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_dead_owner_lock",
            refusal_reason=f"workspace lock file is malformed: {exc}",
        )

    if current is None:
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_dead_owner_lock",
            refusal_reason="no workspace lock present",
        )

    # The dead-owner mode stays distinct from --clear-stale. A
    # timed-out lock has its own canonical recovery path; redirect
    # the operator there.
    if lock.is_timed_out(current):
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_dead_owner_lock",
            refusal_reason=(
                "workspace lock is already timed out; use "
                "`llloom unlock --clear-stale --reason \"...\"` (or "
                "`llloom reconcile`) for timed-out journal-backed "
                "stale locks"
            ),
            prior=current,
            prior_owner_pid_state=local_owner_pid_state(current),
        )

    if current.owner_pid is None or current.owner_pid <= 0:
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_dead_owner_lock",
            refusal_reason=(
                "lock has no recorded owner_pid; --dead-owner requires "
                "local same-host owner metadata"
            ),
            prior=current,
            prior_owner_pid_state=local_owner_pid_state(current),
        )

    current_host = socket.gethostname()
    if not current.owner_hostname or current.owner_hostname != current_host:
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_dead_owner_lock",
            refusal_reason=(
                f"lock owner_hostname={current.owner_hostname!r} does not "
                f"match current host={current_host!r}; --dead-owner is "
                "local same-host only"
            ),
            prior=current,
            prior_owner_pid_state=local_owner_pid_state(current),
        )

    pid_state = local_owner_pid_state(current)
    if pid_state != PID_STATE_DEAD:
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_dead_owner_lock",
            refusal_reason=(
                f"owner_pid_state={pid_state!r}; --dead-owner requires a "
                f"confidently dead local owner process"
            ),
            prior=current,
            prior_owner_pid_state=pid_state,
        )

    if not journal.exists(current.op_id):
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_dead_owner_lock",
            refusal_reason=(
                f"no journal entry for op_id={current.op_id}; "
                "--dead-owner requires a matching in-progress journal "
                "entry as audit evidence"
            ),
            prior=current,
            prior_owner_pid_state=pid_state,
        )

    prior_entry = journal.load(current.op_id)
    if prior_entry.status != "in_progress" or prior_entry.completed_at is not None:
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_dead_owner_lock",
            refusal_reason=(
                f"journal entry for op_id={current.op_id} status="
                f"{prior_entry.status!r} completed_at="
                f"{prior_entry.completed_at!r}; --dead-owner requires "
                "in_progress with no completed_at"
            ),
            prior=current,
            prior_owner_pid_state=pid_state,
        )

    # Race re-read: confirm the lock identity is still what we
    # validated. If the lock was renewed or replaced between our
    # validation and this final read, refuse without mutating anything.
    try:
        recheck = lock.read()
    except LockError as exc:
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_dead_owner_lock",
            refusal_reason=(
                f"workspace lock file became malformed during validation: "
                f"{exc}"
            ),
            prior=current,
            prior_owner_pid_state=pid_state,
        )
    if recheck is None or not _same_lock_identity(current, recheck):
        return _refused(
            target=symbolic_target,
            reason=reason,
            mode="clear_dead_owner_lock",
            refusal_reason=(
                "workspace lock identity changed between validation and "
                "pre-clear re-read; refusing to clear (lock may have been "
                "renewed or replaced)"
            ),
            prior=current,
            prior_owner_pid_state=pid_state,
        )

    # All predicates satisfied — open audit entry, mark prior
    # interrupted, clear the lock, complete the audit.
    op_id = journal.new_op_id(CLEAR_DEAD_OWNER_OP_KIND)
    audit_entry = journal.start(
        op_id=op_id,
        op_kind=CLEAR_DEAD_OWNER_OP_KIND,
        lock_id=current.lock_id,
        planned_writes=[symbolic_target],
    )
    audit_entry.notes.append(
        f"dead_owner_lock reason={reason!r} "
        f"prior_op_id={current.op_id} "
        f"prior_owner_id={current.owner_id} "
        f"prior_owner_pid={current.owner_pid} "
        f"prior_owner_hostname={current.owner_hostname} "
        f"prior_acquired_at={current.acquired_at} "
        f"prior_heartbeat_at={current.heartbeat_at} "
        f"prior_timeout_seconds={current.timeout_seconds} "
        f"owner_pid_state=dead race_recheck=passed"
    )
    journal.mark_interrupted(
        current.op_id,
        note=(
            f"interrupted by unlock --dead-owner {op_id} "
            f"reason={reason!r}"
        ),
    )
    lock.clear()
    journal.complete(
        audit_entry,
        touched_files=[symbolic_target],
        notes=["lock_cleared=true"],
    )

    now = iso_now()
    return UnlockRecord(
        target=symbolic_target,
        reason=reason,
        unlocked_at=now,
        expires_at="",
        op_id=op_id,
        mode="clear_dead_owner_lock",
        lock_cleared=True,
        refused=False,
        prior_op_id=current.op_id,
        prior_owner_id=current.owner_id,
        prior_acquired_at=current.acquired_at,
        prior_heartbeat_at=current.heartbeat_at,
        prior_owner_pid=current.owner_pid,
        prior_owner_hostname=current.owner_hostname,
        prior_owner_pid_state=PID_STATE_DEAD,
    )


def _same_lock_identity(a, b) -> bool:
    """Strict-identity comparison used for the dead-owner race
    re-read. Compares the eight identity fields named in the slice
    contract: ``lock_id``, ``op_id``, ``owner_id``, ``owner_pid``,
    ``owner_hostname``, ``acquired_at``, ``heartbeat_at``,
    ``timeout_seconds``. Any mismatch means the lock was renewed,
    cleared, or replaced between validation and the final read; the
    operation must refuse without mutating anything.
    """
    return (
        a.lock_id == b.lock_id
        and a.op_id == b.op_id
        and a.owner_id == b.owner_id
        and a.owner_pid == b.owner_pid
        and a.owner_hostname == b.owner_hostname
        and a.acquired_at == b.acquired_at
        and a.heartbeat_at == b.heartbeat_at
        and a.timeout_seconds == b.timeout_seconds
    )


def _refused(
    *,
    target: str,
    reason: str,
    mode: str,
    refusal_reason: str,
    prior=None,
    prior_owner_pid_state: str | None = None,
) -> UnlockRecord:
    return UnlockRecord(
        target=target,
        reason=reason,
        unlocked_at=iso_now(),
        expires_at="",
        op_id="",
        mode=mode,
        lock_cleared=False,
        refused=True,
        refusal_reason=refusal_reason,
        prior_op_id=prior.op_id if prior is not None else None,
        prior_owner_id=prior.owner_id if prior is not None else None,
        prior_acquired_at=prior.acquired_at if prior is not None else None,
        prior_heartbeat_at=prior.heartbeat_at if prior is not None else None,
        prior_owner_pid=(
            prior.owner_pid if prior is not None else None
        ),
        prior_owner_hostname=(
            prior.owner_hostname if prior is not None else None
        ),
        prior_owner_pid_state=prior_owner_pid_state,
    )
