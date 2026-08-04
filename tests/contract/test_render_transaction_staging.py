"""Contract: render transaction staging (Slice 074).

Pins the stage-then-commit model that hardens mutating render's
commit boundary. Page bytes and the render-fingerprint snapshot
stage under ``state/transactions/<op_id>/`` and commit together;
an interruption before the commit phase leaves final page bytes
and the stored fingerprint store byte-identical to before; an
interruption during the commit phase leaves a diagnosable
transaction directory on disk while the existing journal/lock
operation semantics keep the in-progress + held signal for
``reconcile``.

Read-only paths from Slice 073 (`--dry-run` and `--list-targets`)
must continue to create no transaction directory.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from llloom.claims.models import Locator
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.reconcile import reconcile
from llloom.ops.render import render
from llloom.state.fingerprints import FingerprintStore
from llloom.state.journal import OperationJournal
from llloom.state.lock import WorkspaceLock
from llloom.state import render_transactions as txn_mod
from llloom.workspace.layout import Workspace


PAGE_TEMPLATE = """\
---
page_id: concept/staging
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.concept.staging -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.staging owner=human -->

Commentary that must survive every render commit.

<!-- /llloom:commentary -->
"""


SOURCE_TEXT = """\
# Article

## Methods

The staging fixture asserts one verifiable claim about the commit model.

Second paragraph irrelevant.
"""


def _seed_workspace(tmp_path: Path) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "alpha.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    page_path = ws.pages / "concepts" / "staging.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    seed = SeedClaim(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        claim_id="c.alpha.1",
        claim_kind="definition",
        claim_text=(
            "The staging fixture asserts one verifiable claim about "
            "the commit model."
        ),
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("concept/staging", "claim_block.concept.staging"),
    )
    result = ingest(
        ws,
        src,
        source_id="src.alpha",
        source_class="markdown_prose",
        seed_claims=[seed],
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    return ws, page_path


def _list_transaction_dirs(ws: Workspace) -> list[Path]:
    if not ws.state_transactions.is_dir():
        return []
    return sorted(p for p in ws.state_transactions.iterdir() if p.is_dir())


def test_state_transactions_directory_is_required(tmp_path: Path) -> None:
    """Workspace.init creates ``state/transactions`` and validation
    refuses a workspace missing it.
    """
    ws = Workspace.init(tmp_path)
    assert ws.state_transactions.is_dir(), (
        "Workspace.init must create state/transactions"
    )
    # Validation refuses a workspace where the directory was removed.
    ws.state_transactions.rmdir()
    from llloom.workspace.layout import WorkspaceError

    with pytest.raises(WorkspaceError) as excinfo:
        Workspace.load(tmp_path)
    assert "state/transactions" in str(excinfo.value)


def test_successful_render_commits_all_staged_outputs(tmp_path: Path) -> None:
    """End-to-end: corrupt the stored fingerprint to drive a real
    commit, run mutating render, assert page + fingerprint commit
    together and the transaction directory is removed on success.
    """
    ws, page_path = _seed_workspace(tmp_path)
    # Capture the rendered page bytes after ingest's initial render.
    initial_page = page_path.read_text(encoding="utf-8")
    fps = FingerprintStore(ws)
    initial_fingerprint = fps.get("concept/staging")
    assert initial_fingerprint is not None

    # Corrupt the stored fingerprint so render has something real to commit.
    fps.set("concept/staging", "sha256:deadbeef")

    # Restore the page to placeholder so the page bytes also change.
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    assert page_path.read_text(encoding="utf-8") != initial_page

    result = render(ws)

    # Page bytes match the post-render snapshot we kept from the
    # initial ingest render.
    assert page_path.read_text(encoding="utf-8") == initial_page
    # Fingerprint advanced back to the canonical value.
    assert fps.get("concept/staging") == initial_fingerprint
    # Result accounting.
    assert any(p.endswith("staging.md") for p in result.rendered_pages), result
    # Transaction directory removed on success.
    assert _list_transaction_dirs(ws) == [], (
        "successful commit must remove the transaction directory"
    )
    # Journal completed and lock released.
    assert not (ws.state_locks / "workspace.yaml").is_file()
    journal = OperationJournal(ws)
    latest = journal.latest()
    assert latest is not None
    assert latest.op_kind == "render"
    assert latest.status == "completed"


def test_pre_commit_failure_leaves_final_state_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monkeypatch ``RenderTransaction.commit`` to raise after staging.
    Final page bytes and stored fingerprints must be byte-identical
    to before the call; the transaction directory remains for
    diagnosis; the journal stays ``in_progress`` and the lock stays
    held so reconcile can recover.
    """
    ws, page_path = _seed_workspace(tmp_path)
    fps = FingerprintStore(ws)
    initial_page = page_path.read_text(encoding="utf-8")
    initial_fingerprints = ws.render_fingerprints.read_text(encoding="utf-8")
    fps.set("concept/staging", "sha256:deadbeef")
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    after_corruption_page = page_path.read_text(encoding="utf-8")
    after_corruption_fingerprints = ws.render_fingerprints.read_text(encoding="utf-8")

    class _SimulatedCommitFailure(RuntimeError):
        pass

    def _failing_commit(self):
        # Mark the manifest as committing before raising — mirroring
        # what the production commit method does at its first line —
        # so the on-disk state is realistic.
        self.write_manifest(status="committing", notes=["simulated commit failure"])
        raise _SimulatedCommitFailure("simulated commit failure")

    monkeypatch.setattr(txn_mod.RenderTransaction, "commit", _failing_commit)

    with pytest.raises(_SimulatedCommitFailure):
        render(ws)

    # Final page bytes byte-identical to the pre-render placeholder
    # we wrote just before the call (the commit step never replaced
    # them with the staged content).
    assert page_path.read_text(encoding="utf-8") == after_corruption_page
    # The stored fingerprint store is byte-identical too.
    assert ws.render_fingerprints.read_text(encoding="utf-8") == after_corruption_fingerprints

    # Transaction directory survives for diagnosis.
    dirs = _list_transaction_dirs(ws)
    assert len(dirs) == 1, dirs
    txn_dir = dirs[0]
    manifest_path = txn_dir / "manifest.yaml"
    assert manifest_path.is_file()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["op_kind"] == "render"
    assert manifest["status"] == "committing"
    assert any("simulated commit failure" in n for n in manifest["notes"])
    # The staged fingerprint file exists inside the txn dir.
    assert (txn_dir / "render_fingerprints.yaml").is_file()
    # The staged page file exists inside the txn dir.
    assert (txn_dir / "pages" / "pages" / "concepts" / "staging.md").is_file()

    # Journal stays in_progress; lock stays held — `operation(...)`
    # contract preserved.
    journal = OperationJournal(ws)
    latest = journal.latest()
    assert latest is not None and latest.op_kind == "render"
    assert latest.status == "in_progress"
    assert (ws.state_locks / "workspace.yaml").is_file()
    # Sanity: the initial values from before we corrupted the
    # fingerprint also hold (we changed the fingerprint twice during
    # the test; the failure path must not have rolled either change
    # forward).
    assert after_corruption_page != initial_page
    assert after_corruption_fingerprints != initial_fingerprints


def test_failure_during_commit_is_diagnosable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a failure while applying staged outputs (a page
    replace mid-loop). The manifest reads ``status: committing`` and
    names the staged paths; the journal stays ``in_progress`` and
    the lock stays held — the existing operation contract carries
    the recovery signal.
    """
    ws, page_path = _seed_workspace(tmp_path)
    FingerprintStore(ws).set("concept/staging", "sha256:deadbeef")
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")

    # Patch Path.replace to fail when called inside the transaction
    # directory commit step. We watch for the specific `.tmp`/`.md`
    # shape staged by the transaction.
    real_replace = Path.replace
    triggered: list[Path] = []

    def _patched_replace(self, target):  # noqa: ARG001
        # Only fail for the page commit (the staged file lives under
        # state/transactions/<op_id>/pages/...). Allow the manifest
        # tmp.replace to succeed so the manifest is still written.
        if "state" in self.parts and "transactions" in self.parts and self.suffix == ".md":
            triggered.append(self)
            raise OSError("simulated mid-commit page replace failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _patched_replace)

    with pytest.raises(OSError):
        render(ws)
    assert triggered, "test setup did not exercise the mid-commit replace path"

    # Manifest survives with a committing status.
    dirs = _list_transaction_dirs(ws)
    assert len(dirs) == 1
    manifest = yaml.safe_load(
        (dirs[0] / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "committing"
    assert manifest["op_kind"] == "render"
    # Journal still in_progress; lock still held.
    journal = OperationJournal(ws)
    latest = journal.latest()
    assert latest is not None and latest.status == "in_progress"
    assert (ws.state_locks / "workspace.yaml").is_file()


def test_dry_run_and_list_targets_create_no_transaction(tmp_path: Path) -> None:
    ws, _ = _seed_workspace(tmp_path)
    FingerprintStore(ws).set("concept/staging", "sha256:deadbeef")
    before = _list_transaction_dirs(ws)
    assert before == []

    result_dry = render(ws, dry_run=True)
    after_dry = _list_transaction_dirs(ws)
    assert after_dry == before, (
        f"dry-run must not create a transaction directory: {after_dry}"
    )
    assert result_dry.dry_run is True
    assert result_dry.plan, "dry-run still reports a plan"

    result_list = render(ws, list_targets=True)
    after_list = _list_transaction_dirs(ws)
    assert after_list == before, (
        f"list-targets must not create a transaction directory: {after_list}"
    )
    assert result_list.list_targets is True


def test_reconcile_clears_abandoned_render_transaction(tmp_path: Path) -> None:
    """Reconcile cleans an abandoned render transaction directory
    only when it is also clearing the matching stale lock (the
    journal-backed recoverability predicate already proved it is
    safe to drop).
    """
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    journal = OperationJournal(ws)
    op_id = journal.new_op_id("render")
    acquired = lock.acquire(op_id=op_id, owner_id="prior-owner", timeout_seconds=1)
    # Backdate heartbeat to mark stale.
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=3600)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = yaml.safe_load(lock.path.read_text(encoding="utf-8"))
    payload["heartbeat_at"] = past
    lock.path.write_text(
        yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
    )
    journal.start(op_id=op_id, op_kind="render", lock_id=acquired.lock_id)

    # Plant an abandoned transaction directory matching the lock's op_id.
    abandoned = ws.state_transactions / op_id
    abandoned.mkdir(parents=True, exist_ok=True)
    (abandoned / "manifest.yaml").write_text(
        "op_id: " + op_id + "\nop_kind: render\nstatus: committing\n",
        encoding="utf-8",
    )

    # Plant an unrelated transaction directory whose op_id does NOT
    # match — reconcile must not touch it.
    unrelated = ws.state_transactions / "op.render.20990101T000000000000Z.99.001"
    unrelated.mkdir()
    (unrelated / "manifest.yaml").write_text(
        "op_id: unrelated\n", encoding="utf-8"
    )

    result = reconcile(ws)
    assert result.lock_cleared is True
    assert op_id in result.journals_marked_interrupted
    assert any(
        f"removed abandoned render transaction directory op_id={op_id}" in a
        for a in result.actions
    ), result.actions

    # Abandoned txn dir gone; unrelated txn dir preserved.
    assert not abandoned.is_dir()
    assert unrelated.is_dir()
