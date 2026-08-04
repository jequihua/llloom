"""`status` operation: compact workspace health summary.

Slice 069 added additive lock-recoverability metadata to
``StatusResult`` so agents can decide between waiting on a live op,
clearing a recoverable stale lock with ``unlock --clear-stale``,
running ``reconcile`` for the automatic-repair path, or escalating
an unrecoverable stale lock for manual investigation. Read access
only; ``status`` still does not mutate the workspace.
"""

from __future__ import annotations

from llloom.claims.store import ClaimStore
from llloom.ops.results import StatusResult
from llloom.sources.registry import SourceRegistry
from llloom.state.journal import OperationJournal
from llloom.state.lock import LockError, WorkspaceLock, local_owner_pid_state
from llloom.workspace.layout import Workspace


def status(workspace: Workspace) -> StatusResult:
    registry = SourceRegistry(workspace)
    store = ClaimStore(workspace)
    lock = WorkspaceLock(workspace)
    journal = OperationJournal(workspace)

    source_count = len(registry.list_ids())
    claim_count = 0
    stale_count = 0
    retracted_count = 0
    for entity in store.iter_entities():
        for assertion in entity.assertions:
            claim_count += 1
            if assertion.status == "stale":
                stale_count += 1
            if assertion.status in {"retracted", "retracted_by_source"}:
                retracted_count += 1

    rendered_page_count = 0
    if workspace.pages.is_dir():
        for page_path in workspace.pages.rglob("*.md"):
            rendered_page_count += 1

    pending_review_count = sum(
        1
        for pid in store.list_proposal_ids()
        if store.load_proposal(pid).status == "pending"
    )

    # Slice 085: additive lock-owner-process diagnostics. ``None`` when
    # no lock is held; otherwise mirror the optional ``Lock.owner_*``
    # fields. ``lock_owner_pid_state`` resolves to ``"alive"`` /
    # ``"dead"`` / ``"unknown"`` when a lock exists; ``None`` when no
    # lock is held. Recoverability / recommended-action fields are
    # still governed by the frozen timeout + journal predicate.
    lock_owner_pid: int | None = None
    lock_owner_hostname: str | None = None
    lock_owner_cwd: str | None = None
    lock_owner_command: str | None = None
    lock_owner_pid_state: str | None = None

    try:
        current_lock = lock.read()
    except LockError as exc:
        current_lock = None
        lock_op_id = None
        lock_acquired_at = None
        lock_heartbeat_at = None
        lock_timeout_seconds = None
        lock_is_timed_out = True
        lock_recoverable = False
        lock_recoverability_reason = f"malformed lock file: {exc}"
        recommended_lock_action = (
            "inspect state/locks/workspace.yaml manually; the file is "
            "unreadable and must be repaired before any operation"
        )
        lock_held = True
        lock_owner = None
    else:
        if current_lock is None:
            lock_held = False
            lock_owner = None
            lock_op_id = None
            lock_acquired_at = None
            lock_heartbeat_at = None
            lock_timeout_seconds = None
            lock_is_timed_out = False
            lock_recoverable = False
            lock_recoverability_reason = None
            recommended_lock_action = None
        else:
            lock_held = True
            lock_owner = current_lock.owner_id
            lock_op_id = current_lock.op_id
            lock_acquired_at = current_lock.acquired_at
            lock_heartbeat_at = current_lock.heartbeat_at
            lock_timeout_seconds = current_lock.timeout_seconds
            lock_is_timed_out = lock.is_timed_out(current_lock)
            lock_owner_pid = current_lock.owner_pid
            lock_owner_hostname = current_lock.owner_hostname
            lock_owner_cwd = current_lock.owner_cwd
            lock_owner_command = current_lock.owner_command
            lock_owner_pid_state = local_owner_pid_state(current_lock)
            if not lock_is_timed_out:
                lock_recoverable = False
                lock_recoverability_reason = "lock has not timed out"
                recommended_lock_action = (
                    f"wait for op_id={current_lock.op_id} owner_id="
                    f"{current_lock.owner_id} or contact the lock owner"
                )
            else:
                recoverable, recovery_reason = lock.is_stale_recoverable(
                    current_lock, journal=journal
                )
                lock_recoverable = recoverable
                lock_recoverability_reason = recovery_reason
                if recoverable:
                    recommended_lock_action = (
                        'llloom unlock --clear-stale --reason "..."  '
                        "(or: llloom reconcile)"
                    )
                else:
                    recommended_lock_action = (
                        f"manual investigation required: "
                        f"{recovery_reason}"
                    )

    last = journal.latest()

    return StatusResult(
        source_count=source_count,
        claim_count=claim_count,
        rendered_page_count=rendered_page_count,
        pending_review_count=pending_review_count,
        stale_count=stale_count,
        retracted_count=retracted_count,
        lock_held=lock_held,
        lock_owner=lock_owner,
        last_operation_id=last.op_id if last is not None else None,
        last_operation_status=last.status if last is not None else None,
        lock_op_id=lock_op_id,
        lock_acquired_at=lock_acquired_at,
        lock_heartbeat_at=lock_heartbeat_at,
        lock_timeout_seconds=lock_timeout_seconds,
        lock_is_timed_out=lock_is_timed_out,
        lock_recoverable=lock_recoverable,
        lock_recoverability_reason=lock_recoverability_reason,
        recommended_lock_action=recommended_lock_action,
        lock_owner_pid=lock_owner_pid,
        lock_owner_hostname=lock_owner_hostname,
        lock_owner_cwd=lock_owner_cwd,
        lock_owner_command=lock_owner_command,
        lock_owner_pid_state=lock_owner_pid_state,
    )
