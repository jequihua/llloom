"""Workspace-scoped file lock with journal-backed stale recovery.

Frozen contract from
``04_specification/storage_and_state_model.md`` Â§Locks:

- single workspace-scoped lock file at ``state/locks/workspace.yaml``
- holder writes owner_id, op_id, timestamps, timeout
- lock is stale-recoverable only if both:
    1. ``heartbeat_at + timeout_seconds`` is in the past, AND
    2. the journal entry for the op has no ``completed_at``
"""

from __future__ import annotations

import errno
import os
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from llloom.workspace.layout import Workspace

if TYPE_CHECKING:
    from llloom.state.journal import OperationJournal


LOCK_FILENAME = "workspace.yaml"
DEFAULT_TIMEOUT_SECONDS = 300

# Bound on the optional ``owner_command`` field. Diagnostic only; not a
# back-channel for source bodies or model payloads.
OWNER_COMMAND_MAX_CHARS = 240

# Returned by :func:`local_owner_pid_state`. Stable string constants so
# tests + ``StatusResult`` / ``DoctorWarning`` evidence can match
# exactly.
PID_STATE_ALIVE = "alive"
PID_STATE_DEAD = "dead"
PID_STATE_UNKNOWN = "unknown"

# Platform dispatch for the liveness probe. Kept as one private module
# constant so deterministic tests can force the POSIX branch without
# monkeypatching the interpreter's own ``os.name``.
_IS_WINDOWS = os.name == "nt"

# Windows liveness-probe constants (query/synchronization rights only; no
# terminate, signal, suspend, debug, write, or broad access rights).
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0x00000000
_WAIT_TIMEOUT = 0x00000102
_ERROR_INVALID_PARAMETER = 87
# Largest representable Windows process identifier (unsigned DWORD). A
# persisted diagnostic PID above this domain is not convertible without
# modulo/truncation, so it is classified ``unknown`` before any API call.
_MAX_WINDOWS_PID = 0xFFFFFFFF


class LockError(Exception):
    """Raised for lock-related failures (conflicts, malformed lock files)."""


@dataclass
class Lock:
    """Lock state persisted in ``state/locks/workspace.yaml``.

    Slice 085 added four optional owner-process metadata fields
    (``owner_pid`` / ``owner_hostname`` / ``owner_cwd`` /
    ``owner_command``) so reviewed diagnostics can surface "the
    local owner process appears dead" without ever bypassing the
    timeout + journal stale-recovery rule. Old lock YAML that
    omits these fields parses cleanly; absence is `None`, not
    malformed.
    """

    lock_id: str
    scope: str
    op_id: str
    owner_id: str
    acquired_at: str
    heartbeat_at: str
    timeout_seconds: int
    owner_pid: int | None = None
    owner_hostname: str | None = None
    owner_cwd: str | None = None
    owner_command: str | None = None

    def to_mapping(self) -> dict:
        # Persist every field, including ``None``s, for a single
        # deterministic on-disk shape that round-trips through
        # ``yaml.safe_dump`` / ``yaml.safe_load``.
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: dict) -> "Lock":
        owner_pid_raw = data.get("owner_pid")
        owner_pid: int | None
        if owner_pid_raw is None:
            owner_pid = None
        else:
            try:
                owner_pid = int(owner_pid_raw)
            except (TypeError, ValueError):
                owner_pid = None
        return cls(
            lock_id=str(data["lock_id"]),
            scope=str(data["scope"]),
            op_id=str(data["op_id"]),
            owner_id=str(data["owner_id"]),
            acquired_at=str(data["acquired_at"]),
            heartbeat_at=str(data["heartbeat_at"]),
            timeout_seconds=int(data["timeout_seconds"]),
            owner_pid=owner_pid,
            owner_hostname=_optional_str(data.get("owner_hostname")),
            owner_cwd=_optional_str(data.get("owner_cwd")),
            owner_command=_optional_str(data.get("owner_command")),
        )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


class WorkspaceLock:
    """File-backed workspace-scope lock.

    The first-slice lock is single-writer. Concurrent ingest attempts
    observe an existing lock and either refuse or (if stale and the
    journal confirms interruption) recover via ``reconcile``.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._path = workspace.state_locks / LOCK_FILENAME

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> Lock | None:
        if not self._path.is_file():
            return None
        try:
            data = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise LockError(f"malformed lock file {self._path}: {exc}") from exc
        if not data:
            return None
        if not isinstance(data, dict):
            raise LockError(
                f"malformed lock file {self._path}: top-level value must be a mapping, "
                f"got {type(data).__name__}"
            )
        try:
            return Lock.from_mapping(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise LockError(
                f"malformed lock file {self._path}: {exc}"
            ) from exc

    def acquire(
        self,
        *,
        op_id: str,
        owner_id: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        owner_pid: int | None = None,
        owner_hostname: str | None = None,
        owner_cwd: str | None = None,
        owner_command: str | None = None,
    ) -> Lock:
        """Acquire the lock exclusively.

        Raises LockError if a live lock is already held. A timed-out
        lock requires reconcile (which consults the journal) before
        acquire can succeed; acquire does not clear locks on its own.

        Slice 085 added four optional owner-process metadata kwargs
        (``owner_pid`` / ``owner_hostname`` / ``owner_cwd`` /
        ``owner_command``) that are persisted on the lock when
        supplied. Callers may omit them; the lock file then carries
        explicit ``null`` values and downstream diagnostics report
        the owner-process state as ``"unknown"``.
        """
        existing = self.read()
        if existing is not None:
            if self.is_timed_out(existing):
                raise LockError(
                    f"workspace lock appears stale; run reconcile before retrying "
                    f"(op_id={existing.op_id}, owner_id={existing.owner_id})"
                )
            raise LockError(
                f"workspace lock held by op_id={existing.op_id} "
                f"owner_id={existing.owner_id}"
            )
        now = _iso_now()
        lock = Lock(
            lock_id="lock.workspace",
            scope="workspace",
            op_id=op_id,
            owner_id=owner_id,
            acquired_at=now,
            heartbeat_at=now,
            timeout_seconds=timeout_seconds,
            owner_pid=owner_pid,
            owner_hostname=owner_hostname,
            owner_cwd=owner_cwd,
            owner_command=(
                owner_command[:OWNER_COMMAND_MAX_CHARS]
                if isinstance(owner_command, str)
                else owner_command
            ),
        )
        self._write(lock)
        return lock

    def heartbeat(self) -> Lock:
        lock = self.read()
        if lock is None:
            raise LockError("no workspace lock to heartbeat")
        lock.heartbeat_at = _iso_now()
        self._write(lock)
        return lock

    def release(self) -> None:
        if self._path.is_file():
            self._path.unlink()

    def clear(self) -> None:
        """Force-clear the lock. Use only from reconcile."""
        self.release()

    def is_timed_out(self, lock: Lock) -> bool:
        """Heartbeat-deadline check only.

        Returns True if ``heartbeat_at + timeout_seconds`` is in the
        past. This is necessary but not sufficient for stale-lock
        recovery; the recovery rule additionally requires that the
        corresponding journal entry has no ``completed_at``. See
        :meth:`is_stale_recoverable`.
        """
        try:
            heartbeat = _parse_iso(lock.heartbeat_at)
        except ValueError:
            return True
        deadline = heartbeat + timedelta(seconds=lock.timeout_seconds)
        return datetime.now(timezone.utc) > deadline

    # Backwards-compatible alias for callers that only need the timeout
    # check. New code should call ``is_timed_out`` for clarity.
    def is_stale(self, lock: Lock) -> bool:  # pragma: no cover - thin alias
        return self.is_timed_out(lock)

    def is_stale_recoverable(
        self,
        lock: Lock,
        *,
        journal: "OperationJournal | None",
    ) -> tuple[bool, str]:
        """Frozen journal-backed stale-recovery rule.

        A lock is stale-recoverable iff:

        1. ``heartbeat_at + timeout_seconds`` is in the past, AND
        2. the corresponding journal entry exists and has no
           ``completed_at``.

        Returns ``(recoverable, reason)``. If not recoverable, ``reason``
        explains why (so ``reconcile`` can report it). The journal
        argument is intentionally accepted as ``None`` for the case where
        a caller has not yet wired the dependency; that case is reported
        as not recoverable rather than silently treated as recoverable.
        """
        if not self.is_timed_out(lock):
            return False, "lock has not timed out"
        if journal is None:
            return False, "journal unavailable"
        if not journal.exists(lock.op_id):
            return False, f"no journal entry for op_id={lock.op_id}"
        entry = journal.load(lock.op_id)
        if entry.completed_at is not None:
            return False, (
                f"journal entry for op_id={lock.op_id} is completed "
                f"(completed_at={entry.completed_at})"
            )
        if entry.status != "in_progress":
            return False, (
                f"journal entry for op_id={lock.op_id} status={entry.status!r}, "
                f"not in_progress"
            )
        return True, "timed out and journal in_progress"

    def _write(self, lock: Lock) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(lock.to_mapping(), sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        tmp.replace(self._path)


def _windows_pid_state(pid: int) -> str:
    """Non-destructive Windows liveness probe for one PID.

    ``os.kill(pid, 0)`` only carries the POSIX "does this process exist"
    guarantee on POSIX. On Windows the CPython implementation does not
    provide that guarantee: depending on version it can terminate the
    process it was meant to inspect, open a terminated process object
    successfully (a false "alive"), or surface an unclassified error for
    an absent PID. Windows therefore uses a read/query-only handle and a
    non-signalling state query instead:

    - a ``pid`` outside the unsigned Windows PID domain
      (``1 <= pid <= 0xFFFFFFFF``; the public helper already covers
      zero/negative values) returns ``unknown`` immediately, before any
      ``ctypes`` conversion or API call — no modulo/truncation/wrap can
      make the probe inspect a different process;
    - ``OpenProcess`` with only ``PROCESS_QUERY_LIMITED_INFORMATION`` and
      ``SYNCHRONIZE`` (the minimum rights needed to observe state);
    - a failed open classifies as ``"dead"`` only for the clearly absent
      case (``ERROR_INVALID_PARAMETER``); access denial and every other
      error are ``"unknown"``;
    - ``WaitForSingleObject(handle, 0)`` performs no signal: a signalled
      (terminated) process object is ``"dead"``, a non-signalled one is
      ``"alive"``, and any query failure is ``"unknown"``;
    - ordinary exceptions raised while loading/configuring the API or
      invoking any of the three calls are bounded to ``"unknown"`` and
      never escape this private boundary;
    - exactly one close attempt follows every acquired handle, from a true
      ``finally`` path, even when the wait raises a propagating
      non-``Exception`` (``KeyboardInterrupt``, ``SystemExit``, or any other
      ``BaseException`` — those are never caught here and still propagate
      after the close attempt); if the close invocation itself fails or
      reports failure during ordinary flow, the overall probe is
      ``"unknown"`` because cleanup was not proven.

    Argument and return types are declared explicitly so handles are not
    truncated on 64-bit Python. Conservative by construction: anything
    that cannot be established safely is ``"unknown"``, never a guessed
    ``"dead"``.
    """
    if pid > _MAX_WINDOWS_PID:
        return PID_STATE_UNKNOWN
    import ctypes
    from ctypes import wintypes

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD, wintypes.BOOL, wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE, False, pid
        )
    except Exception:
        # The API could not be loaded/configured or the open invocation
        # itself failed: no handle was acquired, so there is nothing to
        # close and nothing safe to conclude.
        return PID_STATE_UNKNOWN
    if not handle:
        error = ctypes.get_last_error()
        if error == _ERROR_INVALID_PARAMETER:
            return PID_STATE_DEAD
        return PID_STATE_UNKNOWN
    try:
        try:
            result = kernel32.WaitForSingleObject(handle, 0)
        except Exception:
            # An ordinary wait failure cannot establish state safely.
            result = None
    finally:
        # Exactly one close attempt for the acquired handle on every path,
        # including a propagating KeyboardInterrupt/SystemExit from the
        # wait, which is never caught and continues after this finally.
        try:
            closed = kernel32.CloseHandle(handle)
        except Exception:
            closed = False
    if not closed:
        # Cleanup was not proven; the overall probe cannot stand.
        return PID_STATE_UNKNOWN
    if result is None:
        return PID_STATE_UNKNOWN
    if result == _WAIT_OBJECT_0:
        return PID_STATE_DEAD
    if result == _WAIT_TIMEOUT:
        return PID_STATE_ALIVE
    return PID_STATE_UNKNOWN


def local_owner_pid_state(
    lock: Lock,
    *,
    current_hostname: str | None = None,
) -> str:
    """Classify a lock's owner process from the current machine.

    Slice 085 read-only diagnostic helper. Returns one of the stable
    string constants :data:`PID_STATE_ALIVE`, :data:`PID_STATE_DEAD`,
    or :data:`PID_STATE_UNKNOWN`. Conservative by construction —
    only ``"alive"`` or ``"dead"`` when the local OS check
    confidently says so; ``"unknown"`` otherwise.

    Classification rules:

    - ``"dead"``: the local probe confidently establishes that the PID
      is absent or the process has terminated. On POSIX this is
      ``os.kill(pid, 0)`` raising ``ProcessLookupError`` or ``OSError``
      with ``errno.ESRCH``. On Windows it is the query-only probe in
      :func:`_windows_pid_state` reporting a clearly absent PID or a
      signalled (terminated) process object.
    - ``"alive"``: the local probe confidently establishes that the
      process is currently live. On POSIX this is ``os.kill(pid, 0)``
      returning successfully, OR raising ``PermissionError`` /
      ``OSError`` with ``errno.EPERM`` — the OS is confirming the
      process exists but the current user cannot signal it; a missing
      process would raise ``ProcessLookupError`` / ESRCH instead. On
      Windows it is a successfully opened, non-signalled process.
    - ``"unknown"``: ``owner_pid`` is absent, ``owner_hostname`` is
      absent, no current hostname is available, the hostnames do
      not match, ``owner_pid <= 0``, or the local probe cannot
      establish the state safely — on POSIX an ``OSError`` whose
      ``errno`` is neither ``ESRCH`` nor ``EPERM`` or any other
      unexpected exception; on Windows access denial, an
      unsupported/unexpected error code, or a query failure.

    ``os.kill(pid, 0)`` remains the POSIX branch only: it is the
    accepted POSIX existence probe, and Windows does not share its
    semantics (see :func:`_windows_pid_state`).

    This helper does **not** decide whether to clear a lock. The
    frozen stale-recovery rule (``timeout elapsed + matching
    in-progress journal evidence``) is unchanged. A confidently
    ``"dead"`` owner process surfaces as a doctor warning so the
    operator can wait through the timeout, run ``reconcile``, or
    run ``unlock --clear-stale`` once the rule is satisfied — never
    sooner. The PID-state field is purely diagnostic and never
    grants a lock-clear path.
    """
    if lock.owner_pid is None or lock.owner_hostname is None:
        return PID_STATE_UNKNOWN
    host = current_hostname if current_hostname is not None else socket.gethostname()
    if not host or lock.owner_hostname != host:
        return PID_STATE_UNKNOWN
    pid = lock.owner_pid
    if pid <= 0:
        return PID_STATE_UNKNOWN
    if _IS_WINDOWS:
        return _windows_pid_state(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return PID_STATE_DEAD
    except PermissionError:
        # Process exists but the current user lacks permission to
        # signal it. That's still evidence of "alive" — a missing
        # process would raise ProcessLookupError instead.
        return PID_STATE_ALIVE
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return PID_STATE_DEAD
        if exc.errno == errno.EPERM:
            return PID_STATE_ALIVE
        return PID_STATE_UNKNOWN
    except Exception:  # pragma: no cover - defensive
        return PID_STATE_UNKNOWN
    return PID_STATE_ALIVE


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> datetime:
    # strict: ISO-8601 with trailing Z
    if not value.endswith("Z"):
        raise ValueError(f"expected ISO-8601 UTC timestamp, got {value!r}")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

