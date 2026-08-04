"""Windows live-owner diagnostic safety (F9 / Prompt 032).

Direct regression coverage for the non-destructive Windows liveness probe in
``llloom.state.lock.local_owner_pid_state`` and the diagnostic surfaces that
consume it (``status``, ``doctor``, ``unlock --dead-owner``).

The real-process tests use only disposable children spawned by the test
itself; every child is terminated and reaped in ``finally``. No test ever
targets the test runner, a shell, an agent process, or any pre-existing
process. The mocked Windows-API tests verify rare branches of the probe and
may run on any platform; they never substitute for the real-child tests.
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import llloom.state.lock as lock_module
from llloom.state.journal import OperationJournal
from llloom.state.lock import (
    PID_STATE_ALIVE,
    PID_STATE_DEAD,
    PID_STATE_UNKNOWN,
    WorkspaceLock,
    local_owner_pid_state,
)
from llloom.workspace.layout import Workspace

WINDOWS_ONLY = pytest.mark.skipif(os.name != "nt", reason="Windows liveness contract")

_ERROR_INVALID_PARAMETER = 87
_ERROR_ACCESS_DENIED = 5
_WAIT_OBJECT_0 = 0x00000000
_WAIT_TIMEOUT = 0x00000102
_WAIT_FAILED = 0xFFFFFFFF


def _console_script() -> Path:
    name = "llloom.exe" if os.name == "nt" else "llloom"
    return Path(sys.executable).with_name(name)


def _cli(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(_console_script()), "--root", str(root), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _spawn_disposable_child() -> subprocess.Popen:
    """One disposable Python child that waits without touching any repository."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
    )


def _reap(child: subprocess.Popen) -> None:
    if child.poll() is None:
        child.terminate()
    child.wait(timeout=30)


def _workspace_fingerprint(root: Path) -> tuple:
    """Complete disposable-workspace fingerprint: sorted relative membership,
    entry type, file size, and SHA-256 bytes. Nothing is ignored — state,
    journals, locks, and directories are all covered."""
    import hashlib

    entries = []
    for base, dirs, files in os.walk(root):
        dirs.sort()
        rel_base = os.path.relpath(base, root)
        for name in sorted(dirs):
            rel = os.path.normpath(os.path.join(rel_base, name))
            entries.append((rel, "dir", 0, ""))
        for name in sorted(files):
            full = os.path.join(base, name)
            rel = os.path.normpath(os.path.join(rel_base, name))
            with open(full, "rb") as handle:
                data = handle.read()
            entries.append((rel, "file", len(data), hashlib.sha256(data).hexdigest()))
    return tuple(entries)


def _owner_lock(child_pid: int, op_id: str):
    from llloom.state.lock import Lock

    return Lock(
        lock_id="lock.workspace",
        scope="workspace",
        op_id=op_id,
        owner_id=f"local.{socket.gethostname()}.test",
        acquired_at="2026-08-03T00:00:00Z",
        heartbeat_at="2026-08-03T00:00:00Z",
        timeout_seconds=3600,
        owner_pid=child_pid,
        owner_hostname=socket.gethostname(),
        owner_cwd=None,
        owner_command=None,
    )


def _seed_live_owner_workspace(root: Path, owner_pid: int, op_id: str) -> None:
    """Valid same-host lock plus matching in-progress journal naming the PID."""
    assert _cli(root, "init").returncode == 0
    workspace = Workspace(root)
    lock = WorkspaceLock(workspace)
    acquired = lock.acquire(
        op_id=op_id,
        owner_id=f"local.{socket.gethostname()}.page_create",
        owner_pid=owner_pid,
        owner_hostname=socket.gethostname(),
        owner_cwd=None,
        owner_command="disposable test owner",
    )
    journal = OperationJournal(workspace)
    journal.start(
        op_id=op_id,
        op_kind="page_create",
        lock_id=acquired.lock_id,
        planned_writes=["pages/concepts/x.md"],
    )


@WINDOWS_ONLY
def test_helper_reports_alive_and_a_live_child_survives(tmp_path: Path) -> None:
    """A real disposable child is ``alive`` and is not harmed by the probe."""
    child = _spawn_disposable_child()
    try:
        assert child.poll() is None, "child must be live immediately before the call"
        state = local_owner_pid_state(_owner_lock(child.pid, "op.test.a"))
        assert state == PID_STATE_ALIVE
        assert child.poll() is None, "the probe must not terminate the child"
    finally:
        _reap(child)


@WINDOWS_ONLY
def test_helper_reports_dead_for_a_terminated_child(tmp_path: Path) -> None:
    """A terminated child is classified ``dead`` (signalled process object)."""
    child = _spawn_disposable_child()
    child.terminate()
    child.wait(timeout=30)
    assert local_owner_pid_state(_owner_lock(child.pid, "op.test.b")) == PID_STATE_DEAD


@WINDOWS_ONLY
def test_status_reports_live_owner_and_the_owner_survives(tmp_path: Path) -> None:
    """The installed status surface reads a live owner without harming it."""
    child = _spawn_disposable_child()
    try:
        root = tmp_path / "ws"
        _seed_live_owner_workspace(root, child.pid, "op.page_create.teststatus")
        before = _workspace_fingerprint(root)
        result = _cli(root, "status")
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["lock_held"] is True
        assert payload["lock_op_id"] == "op.page_create.teststatus"
        assert payload["lock_owner_pid_state"] == PID_STATE_ALIVE
        assert payload["lock_owner_pid"] == child.pid
        assert child.poll() is None, "status must not terminate the live owner"
        assert _workspace_fingerprint(root) == before
    finally:
        _reap(child)


@WINDOWS_ONLY
def test_doctor_reports_no_dead_owner_warning_and_the_owner_survives(
    tmp_path: Path,
) -> None:
    """The installed doctor path does not warn dead-owner for a live owner."""
    child = _spawn_disposable_child()
    try:
        root = tmp_path / "ws"
        _seed_live_owner_workspace(root, child.pid, "op.page_create.testdoctor")
        before = _workspace_fingerprint(root)
        result = _cli(root, "doctor")
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        dead_owner_warnings = [
            w for w in payload.get("warnings", [])
            if str(w.get("warning_id", "")).startswith("lock:owner-process-dead")
        ]
        assert dead_owner_warnings == [], dead_owner_warnings
        assert _workspace_fingerprint(root) == before
        assert child.poll() is None, "doctor must not terminate the live owner"
    finally:
        _reap(child)


@WINDOWS_ONLY
def test_dead_owner_unlock_refuses_a_live_owner_safely(tmp_path: Path) -> None:
    """``unlock --dead-owner`` refuses a live owner; workspace fully intact."""
    child = _spawn_disposable_child()
    try:
        root = tmp_path / "ws"
        _seed_live_owner_workspace(root, child.pid, "op.page_create.testunlock")
        before = _workspace_fingerprint(root)
        result = _cli(root, "unlock", "--dead-owner", "--reason", "owner crashed")
        assert result.returncode == 1, (result.stdout, result.stderr)
        assert "Traceback" not in result.stderr
        assert _workspace_fingerprint(root) == before
        status = json.loads(_cli(root, "status").stdout)
        assert status["lock_held"] is True
        assert status["lock_op_id"] == "op.page_create.testunlock"
        assert status["lock_owner_pid_state"] == PID_STATE_ALIVE
        assert child.poll() is None, "the refusal path must not terminate the owner"
    finally:
        _reap(child)


# ---------------------------------------------------------------------------
# F1: explicit Windows PID domain (Prompt 033). No ctypes conversion or API
# call may happen for a positive PID above 0xFFFFFFFF.
# ---------------------------------------------------------------------------


def test_oversized_pid_returns_unknown_without_any_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _forbidden(name, use_last_error=True):
        calls.append(name)
        raise AssertionError("the Windows API must not be loaded for an oversized PID")

    monkeypatch.setattr(ctypes, "WinDLL", _forbidden, raising=False)
    assert lock_module._windows_pid_state(2**32) == PID_STATE_UNKNOWN
    assert calls == []


@WINDOWS_ONLY
def test_wrapped_oversized_pid_does_not_inspect_the_live_child(
    tmp_path: Path,
) -> None:
    """2**32 + child.pid returns unknown and never probes the wrapped child."""
    child = _spawn_disposable_child()
    try:
        assert child.poll() is None
        state = lock_module._windows_pid_state(2**32 + child.pid)
        assert state == PID_STATE_UNKNOWN
        assert child.poll() is None, "the wrapped child must not be inspected"
    finally:
        _reap(child)


@WINDOWS_ONLY
def test_oversized_pid_dead_owner_unlock_refuses_without_workspace_mutation(
    tmp_path: Path,
) -> None:
    """owner_pid=2**32 classifies unknown; the refusal mutates nothing."""
    root = tmp_path / "ws"
    _seed_live_owner_workspace(root, 2**32, "op.page_create.testf1unlock")
    before = _workspace_fingerprint(root)
    result = _cli(root, "unlock", "--dead-owner", "--reason", "owner crashed")
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "Traceback" not in result.stderr
    assert _workspace_fingerprint(root) == before
    status = json.loads(_cli(root, "status").stdout)
    assert status["lock_owner_pid_state"] == PID_STATE_UNKNOWN
    assert status["lock_held"] is True
    assert _workspace_fingerprint(root) == before


# ---------------------------------------------------------------------------
# Mocked Windows-API branch tests. These verify rare branches of the probe
# with a fake kernel32; they never replace the real-child tests above.
# ---------------------------------------------------------------------------


class _FakeKernel32:
    def __init__(self, *, open_handle, last_error, wait_result):
        self.OpenProcess = lambda access, inherit, pid: open_handle
        self.WaitForSingleObject = lambda handle, ms: wait_result
        self.closed = []
        self._last_error = last_error
        handle = open_handle

        def _close(h):
            self.closed.append(h)
            return True

        self.CloseHandle = _close


def _install_fake_kernel32(monkeypatch: pytest.MonkeyPatch, fake: _FakeKernel32) -> None:
    monkeypatch.setattr(
        ctypes, "WinDLL", lambda name, use_last_error=True: fake, raising=False
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: fake._last_error)
    monkeypatch.setattr(lock_module, "_IS_WINDOWS", True)


def test_windows_probe_classifies_a_clearly_absent_pid_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeKernel32(
        open_handle=None, last_error=_ERROR_INVALID_PARAMETER, wait_result=_WAIT_FAILED
    )
    _install_fake_kernel32(monkeypatch, fake)
    assert lock_module._windows_pid_state(12345) == PID_STATE_DEAD
    assert fake.closed == [], "no handle was opened; nothing to close"


def test_windows_probe_returns_unknown_on_access_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeKernel32(
        open_handle=None, last_error=_ERROR_ACCESS_DENIED, wait_result=_WAIT_FAILED
    )
    _install_fake_kernel32(monkeypatch, fake)
    assert lock_module._windows_pid_state(12345) == PID_STATE_UNKNOWN


def test_windows_probe_alive_dead_and_handle_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeKernel32(
        open_handle=4242, last_error=0, wait_result=_WAIT_TIMEOUT
    )
    _install_fake_kernel32(monkeypatch, fake)
    assert lock_module._windows_pid_state(12345) == PID_STATE_ALIVE
    assert fake.closed == [4242], "the opened handle must be closed"

    fake_dead = _FakeKernel32(
        open_handle=4343, last_error=0, wait_result=_WAIT_OBJECT_0
    )
    _install_fake_kernel32(monkeypatch, fake_dead)
    assert lock_module._windows_pid_state(12345) == PID_STATE_DEAD
    assert fake_dead.closed == [4343]


def test_windows_probe_unknown_on_query_failure_and_still_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeKernel32(
        open_handle=4545, last_error=0, wait_result=_WAIT_FAILED
    )
    _install_fake_kernel32(monkeypatch, fake)
    assert lock_module._windows_pid_state(12345) == PID_STATE_UNKNOWN
    assert fake.closed == [4545]


# ---------------------------------------------------------------------------
# F2: bounded Windows API invocation/close failures (Prompt 033). Ordinary
# exceptions become ``unknown``; every acquired handle gets one close attempt.
# The fake callables are plain functions, so the real argtypes/restype
# declaration path stays exercised.
# ---------------------------------------------------------------------------


def _api_raising(*args, **kwargs):
    raise OSError("injected API failure")


class _ExceptionFakeKernel32:
    def __init__(self, *, open_fn, wait_fn, close_fn):
        self.OpenProcess = open_fn
        self.WaitForSingleObject = wait_fn
        self.CloseHandle = close_fn


def test_windows_probe_open_raise_returns_unknown_without_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closes = []
    fake = _ExceptionFakeKernel32(
        open_fn=_api_raising,
        wait_fn=lambda handle, ms: _WAIT_TIMEOUT,
        close_fn=lambda handle: closes.append(handle) or True,
    )
    _install_fake_kernel32(monkeypatch, fake)
    assert lock_module._windows_pid_state(12345) == PID_STATE_UNKNOWN
    assert closes == [], "no handle was acquired; no close may be attempted"


def test_windows_probe_wait_raise_closes_once_and_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closes = []
    fake = _ExceptionFakeKernel32(
        open_fn=lambda access, inherit, pid: 9999,
        wait_fn=_api_raising,
        close_fn=lambda handle: closes.append(handle) or True,
    )
    _install_fake_kernel32(monkeypatch, fake)
    assert lock_module._windows_pid_state(12345) == PID_STATE_UNKNOWN
    assert closes == [9999], "the acquired handle must receive one close attempt"


def test_windows_probe_close_raise_returns_unknown_without_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _ExceptionFakeKernel32(
        open_fn=lambda access, inherit, pid: 8888,
        wait_fn=lambda handle, ms: _WAIT_TIMEOUT,
        close_fn=_api_raising,
    )
    _install_fake_kernel32(monkeypatch, fake)
    assert lock_module._windows_pid_state(12345) == PID_STATE_UNKNOWN


def test_windows_probe_close_reported_failure_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closes = []
    fake = _ExceptionFakeKernel32(
        open_fn=lambda access, inherit, pid: 7777,
        wait_fn=lambda handle, ms: _WAIT_TIMEOUT,
        close_fn=lambda handle: closes.append(handle) or False,
    )
    _install_fake_kernel32(monkeypatch, fake)
    assert lock_module._windows_pid_state(12345) == PID_STATE_UNKNOWN
    assert closes == [7777]


def _interrupting(*args, **kwargs):
    raise KeyboardInterrupt


def test_windows_probe_closes_the_handle_once_when_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F3: a propagating KeyboardInterrupt from the wait still receives
    exactly one close attempt for the acquired handle, and the interrupt
    itself propagates (it is never converted to ``unknown``)."""
    closes = []
    fake = _ExceptionFakeKernel32(
        open_fn=lambda access, inherit, pid: 6666,
        wait_fn=_interrupting,
        close_fn=lambda handle: closes.append(handle) or True,
    )
    _install_fake_kernel32(monkeypatch, fake)
    with pytest.raises(KeyboardInterrupt):
        lock_module._windows_pid_state(12345)
    assert closes == [6666], "the acquired handle must get one close attempt"
