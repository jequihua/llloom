"""Shared helpers used by multiple operations.

Provides:

- operation context manager that acquires/releases the workspace lock
- helpers to construct journal op_ids
- helpers to translate between ``04_specification`` paths and workspace paths
"""

from __future__ import annotations

import os
import socket
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from llloom.state.journal import JournalEntry, OperationJournal
from llloom.state.lock import LockError, OWNER_COMMAND_MAX_CHARS, WorkspaceLock
from llloom.workspace.layout import Workspace


@dataclass
class OperationContext:
    workspace: Workspace
    journal: OperationJournal
    lock: WorkspaceLock
    op_id: str
    entry: JournalEntry


@contextmanager
def operation(
    workspace: Workspace,
    *,
    op_kind: str,
    owner_id: str | None = None,
    planned_writes: list[str] | None = None,
) -> Iterator[OperationContext]:
    """Acquire the workspace lock and open a journal entry.

    Raises LockError if the workspace lock is held by another live op.
    On exception, the journal entry remains ``in_progress`` so that
    ``reconcile`` can observe and classify the interruption.
    """
    journal = OperationJournal(workspace)
    lock = WorkspaceLock(workspace)
    op_id = journal.new_op_id(op_kind)
    actual_owner = owner_id or f"local.{socket.gethostname()}.{op_kind}"

    existing = lock.read()
    if existing is not None:
        if lock.is_timed_out(existing):
            raise LockError(
                "workspace lock appears stale; run `llloom reconcile` first"
            )
        raise LockError(
            f"workspace lock held by op_id={existing.op_id} "
            f"owner_id={existing.owner_id}"
        )

    owner_pid, owner_hostname, owner_cwd, owner_command = _collect_owner_metadata(
        op_kind=op_kind
    )
    acquired = lock.acquire(
        op_id=op_id,
        owner_id=actual_owner,
        owner_pid=owner_pid,
        owner_hostname=owner_hostname,
        owner_cwd=owner_cwd,
        owner_command=owner_command,
    )
    entry = journal.start(
        op_id=op_id,
        op_kind=op_kind,
        lock_id=acquired.lock_id,
        planned_writes=planned_writes or [],
    )
    ctx = OperationContext(
        workspace=workspace,
        journal=journal,
        lock=lock,
        op_id=op_id,
        entry=entry,
    )
    try:
        yield ctx
    except Exception:
        # Leave the journal entry in_progress so reconcile can triage.
        # Do not release the lock on exceptional path either â€” the next
        # run must explicitly reconcile to prove the state is safe.
        raise
    else:
        if ctx.entry.status == "in_progress":
            journal.complete(ctx.entry, touched_files=ctx.entry.touched_files)
        lock.release()


def relative_posix(workspace: Workspace, path: Path) -> str:
    """Return ``path`` as POSIX relative to the workspace root."""
    try:
        rel = path.resolve().relative_to(workspace.root.resolve())
    except ValueError:
        return path.as_posix()
    return rel.as_posix()


def _collect_owner_metadata(
    *, op_kind: str
) -> tuple[int | None, str | None, str | None, str | None]:
    """Best-effort, defensive collection of local owner metadata.

    Slice 085 — populates the optional ``owner_pid`` /
    ``owner_hostname`` / ``owner_cwd`` / ``owner_command`` fields on
    the workspace lock. Any individual field that cannot be obtained
    safely is ``None``; the lock still acquires. The fields are
    diagnostic only — they never govern stale recovery, and
    ``owner_command`` is bounded so the lock file never becomes a
    back-channel for source bodies or model payloads.
    """
    try:
        owner_pid: int | None = int(os.getpid())
        if owner_pid <= 0:
            owner_pid = None
    except Exception:  # pragma: no cover - defensive
        owner_pid = None
    try:
        host = socket.gethostname()
        owner_hostname: str | None = host if host else None
    except Exception:  # pragma: no cover - defensive
        owner_hostname = None
    try:
        owner_cwd: str | None = str(Path.cwd())
    except Exception:  # pragma: no cover - defensive
        owner_cwd = None
    try:
        argv = list(getattr(sys, "argv", []) or [])
        if argv:
            joined = " ".join(str(a) for a in argv)
        else:
            joined = f"op.{op_kind}"
        owner_command: str | None = joined[:OWNER_COMMAND_MAX_CHARS]
    except Exception:  # pragma: no cover - defensive
        owner_command = f"op.{op_kind}"
    return owner_pid, owner_hostname, owner_cwd, owner_command


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

