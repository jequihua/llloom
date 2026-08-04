"""M003/S01 integration tests: no-clobber races, failure matrix, checker, representation.

Every seam is injected deterministically and every race delegates to the real
platform primitive, so a clobbering publication would fail these tests rather
than pass them. No test weakens a bound or oracle to obtain green output; where
a platform genuinely cannot create the entry under test, the case skips with an
explicit reason.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from llloom.ops import page as page_ops
from llloom.ops.page import PageCreateError, create_page
from llloom.state.journal import OperationJournal
from llloom.state.lock import LOCK_FILENAME
from llloom.workspace.layout import Workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "0.1-rc.1"
CRLF = b"\r\n"
PROFILES = [None, CANDIDATE]
PROFILE_IDS = ["legacy", "profiled"]


# ---- helpers ----------------------------------------------------------------


def temp_artifacts(ws: Workspace) -> list[Path]:
    return sorted(ws.pages.rglob("*.tmp"))


def entry_for(ws: Workspace, op_id: str):
    return OperationJournal(ws).load(op_id)


def fingerprint(path: Path) -> tuple:
    """Identity of whatever entry sits at ``path``, file or directory."""
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if path.is_dir():
        return ("dir", sorted(p.name for p in path.iterdir()))
    return ("file", hashlib.sha256(path.read_bytes()).hexdigest())


def assert_clean_refusal(ws: Workspace, result, target: Path, before: tuple) -> None:
    """The exact in-context refusal terminal state after a lost race."""
    assert result.refusal_reason is not None
    assert "already exists" in result.refusal_reason
    assert not result.succeeded
    assert fingerprint(target) == before, "the competitor was modified"
    entry = entry_for(ws, result.op_id)
    assert entry.status == "completed"
    assert entry.touched_files == []
    assert entry.planned_writes == [result.page_path]
    assert not (ws.state_locks / LOCK_FILENAME).exists()
    assert temp_artifacts(ws) == []


class _HandleProxy:
    """Wraps the real file handle so write/flush/close are distinct seams."""

    def __init__(self, handle, fail_on: str, log: list[str]):
        self._handle = handle
        self._fail_on = fail_on
        self._log = log

    def write(self, payload):
        self._log.append("write")
        if self._fail_on == "write":
            raise OSError("injected write failure")
        return self._handle.write(payload)

    def flush(self):
        self._log.append("flush")
        if self._fail_on == "flush":
            raise OSError("injected flush failure")
        return self._handle.flush()

    def close(self):
        self._log.append("close")
        result = self._handle.close()
        if self._fail_on == "close":
            raise OSError("injected close failure")
        return result


def install_handle_seam(monkeypatch, fail_on: str) -> list[str]:
    log: list[str] = []
    real_fdopen = os.fdopen

    def wrapped(fd, *args, **kwargs):
        return _HandleProxy(real_fdopen(fd, *args, **kwargs), fail_on, log)

    monkeypatch.setattr(page_ops.os, "fdopen", wrapped)
    return log


def install_competitor_at_link(monkeypatch, target: Path, payload: bytes | None):
    """Create a competitor immediately before the real atomic primitive runs.

    Delegates to the real :func:`os.link`, so a clobbering implementation
    would overwrite the competitor and fail the caller's assertions.
    """
    real_link = os.link

    def wrapped(src, dst, *args, **kwargs):
        if Path(dst) == target and not os.path.lexists(dst):
            if payload is None:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(page_ops.os, "link", wrapped)


# ---- Correction 1: atomic no-clobber publication races ----------------------


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_competitor_before_publication_wins_unchanged(
    fresh_workspace, monkeypatch, profile
):
    """A regular-file competitor appearing just before the real primitive."""
    ws = fresh_workspace
    target = ws.pages / "concepts" / "race.md"
    payload = b"WINNER-" + (b"PROFILED" if profile else b"LEGACY") + b"\n"
    install_competitor_at_link(monkeypatch, target, payload)

    result = create_page(
        ws, page_id="concept/race", title="Race", framework_profile=profile
    )

    assert target.read_bytes() == payload
    assert_clean_refusal(
        ws, result, target, ("file", hashlib.sha256(payload).hexdigest())
    )


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_directory_competitor_before_publication_wins_unchanged(
    fresh_workspace, monkeypatch, profile
):
    ws = fresh_workspace
    target = ws.pages / "concepts" / "race.md"
    install_competitor_at_link(monkeypatch, target, None)

    result = create_page(
        ws, page_id="concept/race", title="Race", framework_profile=profile
    )

    assert target.is_dir()
    assert_clean_refusal(ws, result, target, ("dir", []))


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_dangling_symlink_competitor_before_publication_wins_unchanged(
    fresh_workspace, monkeypatch, profile
):
    ws = fresh_workspace
    probe = ws.root / "probe-link"
    try:
        probe.symlink_to(ws.root / "nothing")
        probe.unlink()
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform gated
        pytest.skip(f"symlink creation unavailable on this platform: {exc}")

    target = ws.pages / "concepts" / "race.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    real_link = os.link

    def wrapped(src, dst, *args, **kwargs):
        if Path(dst) == target and not os.path.lexists(dst):
            target.symlink_to(ws.root / "no-such-target.md")
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(page_ops.os, "link", wrapped)
    result = create_page(
        ws, page_id="concept/race", title="Race", framework_profile=profile
    )

    assert os.path.lexists(target) and target.is_symlink()
    assert_clean_refusal(ws, result, target, ("symlink", os.readlink(target)))


def test_competitor_created_inside_the_observation_callback_wins(
    fresh_workspace, monkeypatch
):
    """The profiled window between M002 validation and publication."""
    ws = fresh_workspace
    target = ws.pages / "concepts" / "obs.md"
    real_observe = page_ops.observe_page_okf

    def observing(path):
        observation = real_observe(path)
        # The complete temp has now been written and observed; a competitor
        # lands before publication.
        if not os.path.lexists(target):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"WINNER-IN-CALLBACK\n")
        return observation

    monkeypatch.setattr(page_ops, "observe_page_okf", observing)

    result = create_page(
        ws, page_id="concept/obs", title="Obs", framework_profile=CANDIDATE
    )

    assert target.read_bytes() == b"WINNER-IN-CALLBACK\n"
    assert_clean_refusal(
        ws,
        result,
        target,
        ("file", hashlib.sha256(b"WINNER-IN-CALLBACK\n").hexdigest()),
    )


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_competitor_after_the_locked_check_wins(fresh_workspace, monkeypatch, profile):
    """A competitor appearing right after the locked lexists check."""
    ws = fresh_workspace
    target = ws.pages / "concepts" / "post.md"
    real_lexists = os.path.lexists

    def racing(path):
        present = real_lexists(path)
        if Path(path) == target and not present:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"WINNER-POST-CHECK\n")
        return present

    monkeypatch.setattr(page_ops.os.path, "lexists", racing)

    result = create_page(
        ws, page_id="concept/post", title="Post", framework_profile=profile
    )

    assert target.read_bytes() == b"WINNER-POST-CHECK\n"
    assert_clean_refusal(
        ws,
        result,
        target,
        ("file", hashlib.sha256(b"WINNER-POST-CHECK\n").hexdigest()),
    )


def test_publication_never_uses_an_overwrite_capable_primitive(
    fresh_workspace, monkeypatch
):
    """No overwrite-capable primitive may target the final page path.

    Scoped to the publication target: the journal and lock legitimately use
    ``os.replace`` for their own atomic writes elsewhere in the operation.
    """
    ws = fresh_workspace
    target = ws.pages / "concepts" / "safe.md"
    linked: list[Path] = []
    real_link, real_replace, real_path_replace = os.link, os.replace, Path.replace

    def guarded_link(src, dst, *args, **kwargs):
        if Path(dst) == target:
            linked.append(Path(dst))
        return real_link(src, dst, *args, **kwargs)

    def guarded_replace(src, dst, *args, **kwargs):
        assert Path(dst) != target, "publication used os.replace on the page path"
        return real_replace(src, dst, *args, **kwargs)

    def guarded_path_replace(self, dst, *args, **kwargs):
        assert Path(dst) != target, "publication used Path.replace on the page path"
        return real_path_replace(self, dst, *args, **kwargs)

    monkeypatch.setattr(page_ops.os, "link", guarded_link)
    monkeypatch.setattr(page_ops.os, "replace", guarded_replace)
    monkeypatch.setattr(Path, "replace", guarded_path_replace)

    result = create_page(
        ws, page_id="concept/safe", title="Safe", framework_profile=CANDIDATE
    )
    assert (ws.root / result.page_path).exists()
    assert linked == [target], "publication did not use the atomic link primitive"


# ---- Correction 4: the complete failure and cleanup matrix ------------------


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_temp_allocation_failure(fresh_workspace, monkeypatch, profile):
    ws = fresh_workspace

    def failing_mkstemp(*args, **kwargs):
        raise OSError("injected temp allocation failure")

    monkeypatch.setattr(page_ops.tempfile, "mkstemp", failing_mkstemp)
    with pytest.raises(OSError, match="injected temp allocation failure"):
        create_page(ws, page_id="concept/alloc", title="A", framework_profile=profile)
    assert not (ws.pages / "concepts" / "alloc.md").exists()
    assert temp_artifacts(ws) == []


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_handle_acquisition_failure_is_distinct_from_a_write_failure(
    fresh_workspace, monkeypatch, profile
):
    ws = fresh_workspace
    log: list[str] = []
    def failing(fd, *args, **kwargs):
        # Model a genuine fdopen failure: the descriptor is NOT consumed, so
        # the production path still owns and must close it.
        log.append("fdopen")
        raise OSError("injected handle acquisition failure")

    monkeypatch.setattr(page_ops.os, "fdopen", failing)
    with pytest.raises(OSError, match="injected handle acquisition failure"):
        create_page(ws, page_id="concept/hnd", title="H", framework_profile=profile)

    assert log == ["fdopen"], "no write was attempted"
    assert not (ws.pages / "concepts" / "hnd.md").exists()
    assert temp_artifacts(ws) == []


@pytest.mark.parametrize("seam", ["write", "flush", "close"])
@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_write_flush_and_close_are_distinct_seams(
    fresh_workspace, monkeypatch, profile, seam
):
    ws = fresh_workspace
    log = install_handle_seam(monkeypatch, seam)

    with pytest.raises(OSError, match=f"injected {seam} failure"):
        create_page(ws, page_id="concept/seam", title="S", framework_profile=profile)

    # The failing seam was reached, and every earlier seam ran first.
    order = ["write", "flush", "close"]
    cut = order.index(seam) + 1
    assert log[:cut] == order[:cut]
    assert not (ws.pages / "concepts" / "seam.md").exists()
    assert temp_artifacts(ws) == []


def test_m002_mismatch_publishes_nothing(fresh_workspace, monkeypatch):
    ws = fresh_workspace

    class NotR14:
        read_status = "ok"
        okf_concept_result = "pass"
        okf_concept_reason = None
        framework_profile_result = "fail"
        framework_profile_reason = "PROFILE_VERSION_UNSUPPORTED"
        declared_framework_profile = "9.9-rc.9"
        execution_eligibility = "not_evaluated"

    def observing(path):
        assert Path(path).read_bytes().startswith(b"---\ntype: page\n")
        return NotR14()

    monkeypatch.setattr(page_ops, "observe_page_okf", observing)
    with pytest.raises(PageCreateError) as excinfo:
        create_page(ws, page_id="concept/bad", title="B", framework_profile=CANDIDATE)

    assert "did not observe as the exact accepted profile outcome" in str(excinfo.value)
    assert "framework_profile_result='fail'" in str(excinfo.value)
    assert not (ws.pages / "concepts" / "bad.md").exists()
    assert temp_artifacts(ws) == []


def test_m002_exception_publishes_nothing(fresh_workspace, monkeypatch):
    ws = fresh_workspace
    monkeypatch.setattr(
        page_ops,
        "observe_page_okf",
        lambda path: (_ for _ in ()).throw(RuntimeError("injected observer crash")),
    )
    with pytest.raises(RuntimeError, match="injected observer crash"):
        create_page(ws, page_id="concept/crash", title="C", framework_profile=CANDIDATE)
    assert not (ws.pages / "concepts" / "crash.md").exists()
    assert temp_artifacts(ws) == []


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_non_collision_publication_failure_preserves_a_competing_target(
    fresh_workspace, monkeypatch, profile
):
    """A transfer failure that is not a collision, with a target present."""
    ws = fresh_workspace
    target = ws.pages / "concepts" / "pub.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    real_link = os.link

    def failing_link(src, dst, *args, **kwargs):
        if Path(dst) == target:
            # A competitor exists and the primitive fails for an unrelated
            # reason, so this is not the collision path.
            target.write_bytes(b"COMPETITOR\n")
            raise OSError(5, "injected non-collision publication failure")
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(page_ops.os, "link", failing_link)

    with pytest.raises(PageCreateError, match="atomic no-clobber link primitive"):
        create_page(ws, page_id="concept/pub", title="P", framework_profile=profile)

    assert target.read_bytes() == b"COMPETITOR\n"
    assert temp_artifacts(ws) == []


def test_owned_temp_cleanup_failure_is_surfaced_not_swallowed(
    fresh_workspace, monkeypatch
):
    ws = fresh_workspace
    monkeypatch.setattr(
        page_ops,
        "observe_page_okf",
        lambda path: (_ for _ in ()).throw(
            PageCreateError("primary validation failure")
        ),
    )
    real_unlink = Path.unlink

    def denied(self, *args, **kwargs):
        if self.suffix == ".tmp":
            raise PermissionError("injected cleanup denial")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", denied)

    with pytest.raises(PageCreateError) as excinfo:
        create_page(ws, page_id="concept/deny", title="D", framework_profile=CANDIDATE)

    message = str(excinfo.value)
    # The primary failure context survives and the cleanup failure is explicit.
    assert "primary validation failure" in message
    assert "could not be removed" in message
    assert "retained" in message
    # Bounded: no traceback and no machine-local temp path leaked.
    assert "Traceback" not in message
    assert str(ws.root) not in message
    assert ".tmp" not in message
    # Honest terminal state: the owned artifact really is retained.
    assert len(temp_artifacts(ws)) == 1
    assert not (ws.pages / "concepts" / "deny.md").exists()


def test_nested_parent_created_by_this_operation_is_removed_on_failure(
    fresh_workspace, monkeypatch
):
    ws = fresh_workspace
    monkeypatch.setattr(
        page_ops,
        "observe_page_okf",
        lambda path: (_ for _ in ()).throw(PageCreateError("injected")),
    )
    with pytest.raises(PageCreateError, match="injected"):
        create_page(
            ws,
            page_id="concept/new-parent/fail",
            title="F",
            framework_profile=CANDIDATE,
        )

    assert not (ws.pages / "concepts" / "new-parent").exists()
    assert (ws.pages / "concepts").is_dir(), "the class root is never removed"
    assert temp_artifacts(ws) == []


def test_a_parent_filled_by_concurrent_work_is_preserved_on_failure(
    fresh_workspace, monkeypatch
):
    ws = fresh_workspace
    stranger = ws.pages / "concepts" / "shared-parent" / "other.md"

    def observing(path):
        stranger.parent.mkdir(parents=True, exist_ok=True)
        stranger.write_bytes(b"CONCURRENT\n")
        raise PageCreateError("injected")

    monkeypatch.setattr(page_ops, "observe_page_okf", observing)
    with pytest.raises(PageCreateError, match="injected"):
        create_page(
            ws,
            page_id="concept/shared-parent/fail",
            title="F",
            framework_profile=CANDIDATE,
        )

    assert stranger.read_bytes() == b"CONCURRENT\n", "another writer's content survived"
    assert temp_artifacts(ws) == []


def test_pre_existing_parent_is_never_removed_on_failure(fresh_workspace, monkeypatch):
    ws = fresh_workspace
    parent = ws.pages / "concepts" / "kept"
    parent.mkdir(parents=True)
    monkeypatch.setattr(
        page_ops,
        "observe_page_okf",
        lambda path: (_ for _ in ()).throw(PageCreateError("injected")),
    )
    with pytest.raises(PageCreateError, match="injected"):
        create_page(
            ws, page_id="concept/kept/fail", title="F", framework_profile=CANDIDATE
        )
    assert parent.is_dir(), "a pre-existing directory must not be removed"


# ---- pre-existing entries at the output path (pre-operation refusals) -------


def test_existing_regular_file_refuses_without_mutation(fresh_workspace):
    ws = fresh_workspace
    target = ws.pages / "concepts" / "taken.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"ORIGINAL\n")
    before = fingerprint(target)
    with pytest.raises(PageCreateError, match="already exists"):
        create_page(ws, page_id="concept/taken", title="T", framework_profile=CANDIDATE)
    assert fingerprint(target) == before
    assert temp_artifacts(ws) == []


def test_existing_directory_refuses_without_mutation(fresh_workspace):
    ws = fresh_workspace
    target = ws.pages / "concepts" / "dir.md"
    target.mkdir(parents=True)
    (target / "sentinel.txt").write_bytes(b"KEEP\n")
    with pytest.raises(PageCreateError, match="already exists"):
        create_page(ws, page_id="concept/dir", title="D", framework_profile=CANDIDATE)
    assert (target / "sentinel.txt").read_bytes() == b"KEEP\n"
    assert temp_artifacts(ws) == []


def test_unrelated_temp_like_file_is_never_reused_or_removed(fresh_workspace):
    ws = fresh_workspace
    concepts = ws.pages / "concepts"
    concepts.mkdir(parents=True, exist_ok=True)
    stranger = concepts / "shared.md.tmp"  # the pre-M003 shared temp name
    stranger.write_bytes(b"NOT MINE\n")
    before = fingerprint(stranger)

    result = create_page(
        ws, page_id="concept/shared", title="Shared", framework_profile=CANDIDATE
    )

    assert (ws.root / result.page_path).exists()
    assert fingerprint(stranger) == before


# ---- journal and residue terminal states ------------------------------------


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_success_has_one_complete_journal_entry_and_no_residue(fresh_workspace, profile):
    ws = fresh_workspace
    result = create_page(
        ws, page_id="concept/done", title="Done", framework_profile=profile
    )
    entry = entry_for(ws, result.op_id)
    assert entry.op_kind == "page_create"
    assert entry.status == "completed"
    assert entry.planned_writes == ["pages/concepts/done.md"]
    assert entry.touched_files == ["pages/concepts/done.md"]
    assert not (ws.state_locks / LOCK_FILENAME).exists()
    assert temp_artifacts(ws) == []


def test_in_context_failure_retains_accepted_journal_semantics(
    fresh_workspace, monkeypatch
):
    """An in-context exception leaves the entry in_progress for reconcile.

    This is the repository's accepted ``src/llloom/ops/_context.py`` contract:
    on the exceptional path the journal entry stays in_progress and the lock is
    deliberately retained so the next run must reconcile. This slice does not
    change it; the exact terminal state is asserted rather than merely checking
    that a journal file exists.
    """
    ws = fresh_workspace
    monkeypatch.setattr(
        page_ops,
        "observe_page_okf",
        lambda path: (_ for _ in ()).throw(PageCreateError("injected")),
    )
    with pytest.raises(PageCreateError, match="injected"):
        create_page(ws, page_id="concept/half", title="H", framework_profile=CANDIDATE)

    entries = sorted(ws.state_journals.rglob("op.page_create.*"))
    assert len(entries) == 1
    entry = OperationJournal(ws).load(entries[0].stem)
    assert entry.status == "in_progress"
    assert entry.planned_writes == ["pages/concepts/half.md"]
    assert entry.touched_files == []
    assert not (ws.pages / "concepts" / "half.md").exists()
    assert temp_artifacts(ws) == []
    assert (ws.state_locks / LOCK_FILENAME).exists()


# ---- Correction 5: the independent template checker lane -------------------


def _template_root() -> Path | None:
    proc = subprocess.run(
        ["git", "config", "--local", "--get", "handoff.templatebaseline"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    root = proc.stdout.strip()
    if not root:
        return None
    checker = Path(root) / "scripts" / "artifact_integrity_preflight.py"
    return Path(root) if checker.is_file() else None


def test_configured_public_checker_lane_agrees_exactly_with_m002(
    fresh_workspace, tmp_path
):
    template_root = _template_root()
    if template_root is None:  # pragma: no cover - environment gated
        pytest.skip("configured template baseline checker lane is not available")

    ws = fresh_workspace
    result = create_page(
        ws, page_id="concept/checked", title="Checked", framework_profile=CANDIDATE
    )
    generated = (ws.root / result.page_path).read_bytes()
    generated_digest = hashlib.sha256(generated).hexdigest()

    # Disposable, digest-verified copy. The template baseline is never written.
    probe_root = tmp_path / "checker-root"
    probe_root.mkdir()
    copied = probe_root / "candidate.md"
    copied.write_bytes(generated)
    assert copied.read_bytes() == generated
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == generated_digest

    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(template_root / "scripts" / "artifact_integrity_preflight.py"),
            "--root",
            str(probe_root),
            "--profile",
            "--json",
            "candidate.md",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)

    assert report["schema_version"] == "template.okf_profile_check.v2"
    assert report["errors"] == 0
    assert report["warnings"] == 0
    assert report["findings"] == []
    assert len(report["artifacts"]) == 1
    record = report["artifacts"][0]
    assert record["path"] == "candidate.md"
    assert record["okf_concept"] == {"result": "pass", "reason": None}
    assert record["framework_profile"] == {"result": "pass", "reason": None}
    assert record["execution_eligibility"] == "not_evaluated"

    # The checker agrees with M002 on the same bytes, and the copy is unchanged.
    observation = page_ops.observe_page_okf(ws.root / result.page_path)
    assert observation.okf_concept_result == "pass"
    assert observation.framework_profile_result == "pass"
    assert observation.execution_eligibility == "not_evaluated"
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == generated_digest


# ---- Correction 5: causal LF/CRLF source-representation control -------------


SUBPROCESS_PROBE = '''\
import hashlib, json, os, sys
root, ws_root, target_root = sys.argv[1], sys.argv[2], sys.argv[3]

def resolved(path):
    return os.path.normcase(os.path.realpath(path))

def under(path, ancestor):
    """True only when `path` resolves inside `ancestor`, by path ancestry.

    Deliberately not a name-substring test: an external basetemp whose own
    directory name happens to contain the repository's name is not the
    repository, and a copied source tree under it must not be misclassified as
    the configured checkout.
    """
    path, ancestor = resolved(path), resolved(ancestor)
    return path == ancestor or path.startswith(ancestor + os.sep)

# The bootstrap interpreter may carry an editable install of the target checkout.
# Drop exactly the entries that resolve inside the configured target repository
# root, so the only llloom on the path is this disposable copy.
sys.path = [p for p in sys.path if not (p and under(p, target_root))]
sys.path.insert(0, os.path.join(root, "src"))
import llloom.ops.page as page_mod
from llloom.workspace.layout import Workspace
from llloom.ops.page import create_page
inside = os.path.join(root, "src")
report = {
    "from_this_root": under(page_mod.__file__, inside),
    "sys_path_in_repo": [p for p in sys.path if p and under(p, target_root)],
}
ws = Workspace.init(ws_root)
result = create_page(ws, page_id="concept/rep", title="Rep", framework_profile="0.1-rc.1")
raw = open(os.path.join(ws_root, result.page_path), "rb").read()
report["digest"] = hashlib.sha256(raw).hexdigest()
report["length"] = len(raw)
report["crlf"] = raw.count(bytes([13, 10]))
print(json.dumps(report))
'''


def run_representation_probe(work_root: Path, label: str, newline: bytes) -> dict:
    """Copy the product source into ``work_root``, set its representation, probe it.

    ``work_root`` may be any writable directory, including one whose own name
    contains the repository name. Provenance is decided by resolved path
    ancestry against the real configured checkout, never by how a temporary
    directory happens to be spelled.
    """
    work_root.mkdir(parents=True, exist_ok=True)
    probe = work_root / "probe.py"
    probe.write_text(SUBPROCESS_PROBE, encoding="utf-8")

    root = work_root / f"src-{label}"
    (root / "src").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "src" / "llloom", root / "src" / "llloom")

    # Rewrite the product source module in the chosen representation.
    module = root / "src" / "llloom" / "ops" / "page.py"
    body = module.read_bytes().replace(CRLF, b"\n")
    module.write_bytes(body if newline == b"\n" else body.replace(b"\n", CRLF))
    assert (CRLF in module.read_bytes()) is (newline == CRLF)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [
            sys.executable, "-B", str(probe), str(root),
            str(work_root / f"ws-{label}"), str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(work_root),
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_lf_and_crlf_source_representations_emit_identical_profiled_bytes(tmp_path):
    """Causal control: vary the representation of the product source itself."""
    reports = {
        label: run_representation_probe(tmp_path, label, newline)
        for label, newline in (("lf", b"\n"), ("crlf", CRLF))
    }

    for label, report in reports.items():
        assert report["from_this_root"], f"{label} imported page.py from elsewhere"
        assert report["sys_path_in_repo"] == [], f"{label} had the target repo on sys.path"
        assert report["crlf"] == 0, f"{label} emitted CRLF in the profiled page"

    assert reports["lf"]["digest"] == reports["crlf"]["digest"]
    assert reports["lf"]["length"] == reports["crlf"]["length"]


@pytest.mark.parametrize("prefix", ["m003_neutral_", "m003_llloomokf-dev_"],
                         ids=["neutral_basetemp", "adversarial_basetemp"])
def test_source_provenance_does_not_depend_on_the_basetemp_name(prefix):
    """The probe must survive an external work root named after the repository.

    The adversarial case is the exact reproducibility defect the M005/S01 review
    recorded: a substring filter classified the probe's own copied source tree as
    the configured checkout purely because an ancestor directory was spelled
    ``llloomokf-dev``. Resolved ancestry against the real checkout cannot.
    """
    work_root = Path(tempfile.mkdtemp(prefix=prefix))
    assert not str(work_root.resolve()).startswith(str(REPO_ROOT.resolve())), (
        "the adversarial work root must live outside the configured checkout"
    )
    try:
        report = run_representation_probe(work_root, "lf", b"\n")
        assert report["from_this_root"]
        assert report["sys_path_in_repo"] == []
        assert report["crlf"] == 0
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    assert not work_root.exists()


def test_generated_profiled_bytes_are_identical_across_repeated_workspaces(tmp_path):
    digests = set()
    for index in range(3):
        ws = Workspace.init(tmp_path / f"repeat{index}")
        result = create_page(
            ws, page_id="concept/rep", title="Rep", framework_profile=CANDIDATE
        )
        digests.add(
            hashlib.sha256((ws.root / result.page_path).read_bytes()).hexdigest()
        )
    assert len(digests) == 1


# ---- Prompt 016 / F7: post-publication cleanup denial -----------------------


def test_post_publication_owned_temp_cleanup_denial_leaves_a_complete_reconcile_visible_state(
    fresh_workspace, monkeypatch
):
    """The branch after a *successful* atomic publication, not before it.

    The real ``os.link`` runs and makes the final page visible; only the
    subsequent deletion of this operation's own temp name is denied. This is
    causally distinct from the pre-publication cleanup-denial test above,
    which fails M002 observation first and therefore never publishes.
    """
    ws = fresh_workspace
    page_relative = "pages/concepts/pub-parent/done.md"
    target = ws.root / page_relative
    parent = target.parent

    # Path-keyed fingerprint over EVERY regular file in the workspace, including
    # the overview page, schema, policy, manifest, and render-fingerprint files.
    # A path set alone cannot prove a pre-existing file was left unchanged.
    def workspace_fingerprints() -> dict[str, tuple[str, int, int]]:
        out: dict[str, tuple[str, int, int]] = {}
        for candidate in ws.root.rglob("*"):
            if candidate.is_file():
                stat = candidate.stat()
                out[candidate.relative_to(ws.root).as_posix()] = (
                    hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    stat.st_size,
                    stat.st_mtime_ns,
                )
        return out

    before_state = workspace_fingerprints()
    assert before_state, "the disposable workspace must contain pre-existing files"

    published: list[Path] = []
    real_link = os.link
    real_unlink = Path.unlink

    def recording_link(src, dst, *args, **kwargs):
        result = real_link(src, dst, *args, **kwargs)
        published.append(Path(dst))
        return result

    def denied_unlink(self, *args, **kwargs):
        # Deny only this operation's temp name, and only once publication
        # has genuinely happened.
        if published and self.suffix == ".tmp":
            raise PermissionError("injected post-publication cleanup denial")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(page_ops.os, "link", recording_link)
    monkeypatch.setattr(Path, "unlink", denied_unlink)

    with pytest.raises(PageCreateError) as excinfo:
        create_page(
            ws,
            page_id="concept/pub-parent/done",
            title="Done",
            framework_profile=CANDIDATE,
        )

    # The real publication primitive ran and succeeded before the denial.
    assert published == [target]

    # 1. The bounded post-publication error, distinct from the pre-publication one.
    message = str(excinfo.value)
    assert f"published {page_relative!r}" in message
    assert "could not remove this operation's" in message
    assert "the page is complete" in message
    assert "reconciliation" in message
    assert "Traceback" not in message
    assert str(ws.root) not in message
    assert str(target) not in message

    # 2. The final page exists as a regular file with the exact profiled bytes.
    assert target.is_file()
    expected = (
        "---\n"
        "type: page\n"
        'framework_profile: "0.1-rc.1"\n'
        "page_id: concept/pub-parent/done\n"
        "page_class: concept\n"
        "write_policy: mixed\n"
        "status: draft\n"
        "---\n"
        "\n"
        "# Done\n"
        "\n"
        "<!-- llloom:claim-block id=claim_block.concept.pub-parent.done -->\n"
        "\n"
        "<!-- /llloom:claim-block -->\n"
        "\n"
        "<!-- llloom:commentary id=commentary.concept.pub-parent.done owner=human -->\n"
        "\n"
        "<!-- /llloom:commentary -->\n"
    ).encode("utf-8")
    final_bytes = target.read_bytes()
    assert final_bytes == expected
    assert CRLF not in final_bytes
    assert final_bytes.endswith(b"\n") and not final_bytes.endswith(b"\n\n")

    # 3. The published page independently observes as the exact R14 tuple.
    observation = page_ops.observe_page_okf(target)
    assert (
        observation.read_status,
        observation.okf_concept_result,
        observation.okf_concept_reason,
        observation.framework_profile_result,
        observation.framework_profile_reason,
        observation.execution_eligibility,
        observation.declared_framework_profile,
    ) == ("ok", "pass", None, "pass", None, "not_evaluated", "0.1-rc.1")

    # 4. Exactly one owned temp remains beside the page, holding the same bytes.
    retained = temp_artifacts(ws)
    assert len(retained) == 1
    assert retained[0].parent == parent
    assert retained[0].read_bytes() == final_bytes

    # 5. Exactly one page-create journal entry, in_progress, with empty touched_files.
    entries = sorted(ws.state_journals.rglob("op.page_create.*"))
    assert len(entries) == 1
    entry = OperationJournal(ws).load(entries[0].stem)
    assert entry.op_kind == "page_create"
    assert entry.status == "in_progress"
    assert entry.planned_writes == [page_relative]
    assert entry.touched_files == []

    # 6. The lock is retained so the next run must reconcile.
    assert (ws.state_locks / LOCK_FILENAME).exists()

    # 7. The operation-created parent holds exactly the page and the retained temp.
    assert parent.is_dir()
    assert sorted(p.name for p in parent.iterdir()) == sorted(
        [target.name, retained[0].name]
    )

    # 8. Nothing unrelated was created, removed, or changed. Three independent
    #    properties, none of which is derived from product output.
    after_state = workspace_fingerprints()

    # 8a. Exactly the four expected additions.
    assert set(after_state) - set(before_state) == {
        page_relative,
        retained[0].relative_to(ws.root).as_posix(),
        entries[0].relative_to(ws.root).as_posix(),
        (ws.state_locks / LOCK_FILENAME).relative_to(ws.root).as_posix(),
    }

    # 8b. No pre-existing path was removed.
    assert set(before_state) - set(after_state) == set()

    # 8c. Every pre-existing file keeps its complete (sha256, size, mtime_ns).
    unchanged = {path: before_state[path] for path in before_state}
    assert {path: after_state[path] for path in before_state} == unchanged
