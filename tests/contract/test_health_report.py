"""Contract tests for the deepened ``rebuild health_report`` surface.

The deepened report extends the existing
``rebuild(workspace, target="health_report")`` target into a
deterministic read-only drift surface over current workspace state.
The function is detection-only: it does not remediate, does not clear
locks, does not rebuild missing sidecars, and does not mark journals
interrupted. It writes only to the existing derived report path
``state/reports/health/health_report.yaml`` and to its own completed
journal entry.

One contract test per drift field, plus a clean-workspace control.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from llloom.claims.models import Assertion, EntityContainer, RenderTarget
from llloom.claims.store import ClaimStore
from llloom.ops import HealthReport
from llloom.ops.rebuild import rebuild
from llloom.pages.render import compute_render_fingerprint
from llloom.sources.registry import SourceRegistry
from llloom.state.fingerprints import FingerprintStore
from llloom.state.journal import JournalEntry, OperationJournal
from llloom.state.lock import WorkspaceLock
from llloom.structured.extract import (
    StructureItem,
    StructureReport,
    write_structure_report,
)
from llloom.workspace.layout import Workspace


def _backdate_lock_heartbeat(lock: WorkspaceLock, seconds_ago: int = 3600) -> None:
    """Pull the heartbeat back so ``is_timed_out`` returns True."""
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = yaml.safe_load(lock.path.read_text(encoding="utf-8"))
    payload["heartbeat_at"] = past
    lock.path.write_text(
        yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
    )


def test_health_report_lists_interrupted_journals(tmp_path: Path) -> None:
    """An in_progress journal entry whose op_id is not claimed by any
    live lock surfaces as a reconcile candidate."""
    ws = Workspace.init(tmp_path)
    journal = OperationJournal(ws)
    abandoned_op_id = journal.new_op_id("ingest")
    journal.start(
        op_id=abandoned_op_id,
        op_kind="ingest",
        lock_id="lock.workspace",
    )

    report = rebuild(ws, target="health_report")

    assert isinstance(report, HealthReport)
    assert report.target == "health_report"
    assert abandoned_op_id in report.interrupted_journals
    # The rebuild's own journal entry is completed, never interrupted.
    assert not any(
        oid.startswith("op.rebuild.health_report.")
        for oid in report.interrupted_journals
    )


def test_health_report_flags_stale_lock_unrecoverable(tmp_path: Path) -> None:
    """A timed-out lock with no in-progress journal cannot be cleared by
    reconcile; ``stale_lock_unrecoverable`` flags this exactly."""
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    lock.acquire(op_id="op.ingest.20990101T000000Z", owner_id="t", timeout_seconds=1)
    _backdate_lock_heartbeat(lock)

    report = rebuild(ws, target="health_report")

    assert report.stale_lock_unrecoverable is True
    assert report.lock_held is True
    assert report.lock_owner == "t"


def test_health_report_reports_missing_search_sidecar(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    assert not ws.search_db.is_file()

    report = rebuild(ws, target="health_report")

    assert report.search_sidecar == "missing"


def test_health_report_reports_missing_graph_sidecar(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    assert not ws.graph_db.is_file()

    report = rebuild(ws, target="health_report")

    assert report.graph_sidecar == "missing"


def test_health_report_reports_structure_report_drift(tmp_path: Path) -> None:
    """A registered structured source whose on-disk report carries a
    ``content_hash`` that no longer matches the source registry record
    surfaces as drift."""
    ws = Workspace.init(tmp_path)
    raw_path = ws.raw_sources / "policies.yaml"
    raw_path.write_text("policies:\n  markdown_prose: claim_extract\n", encoding="utf-8")

    registry = SourceRegistry(ws)
    record, state = registry.register(
        source_id="src.policies",
        raw_path=raw_path,
        source_class="structured_yaml",
    )
    assert state == "new"

    drifted_report = StructureReport(
        source_id=record.source_id,
        source_class="structured_yaml",
        locator_type="code_v1",
        content_hash="sha256:" + "0" * 64,
        language="yaml",
        items=[
            StructureItem(
                kind="mapping_key",
                name="policies",
                symbol_path="policies",
                locator={
                    "locator_type": "code_v1",
                    "path": "raw/sources/policies.yaml",
                    "start_line": 1,
                    "start_col": 1,
                    "end_line": 1,
                    "end_col": 9,
                },
            )
        ],
    )
    write_structure_report(ws, drifted_report)
    assert ws.structure_report_path(record.source_id).is_file()
    assert drifted_report.content_hash != record.content_hash

    report = rebuild(ws, target="health_report")

    assert record.source_id in report.structure_report_drift
    assert record.source_id not in report.missing_structure_reports


def test_health_report_reports_missing_structure_reports(tmp_path: Path) -> None:
    """A registered source whose policy is ``structure_extract`` but
    whose report file is absent surfaces in ``missing_structure_reports``,
    not in ``structure_report_drift``."""
    ws = Workspace.init(tmp_path)
    raw_path = ws.raw_sources / "policies.yaml"
    raw_path.write_text("policies:\n  markdown_prose: claim_extract\n", encoding="utf-8")

    registry = SourceRegistry(ws)
    record, _ = registry.register(
        source_id="src.policies",
        raw_path=raw_path,
        source_class="structured_yaml",
    )
    assert not ws.structure_report_path(record.source_id).is_file()

    report = rebuild(ws, target="health_report")

    assert record.source_id in report.missing_structure_reports
    assert record.source_id not in report.structure_report_drift


def test_health_report_reports_render_fingerprint_drift(tmp_path: Path) -> None:
    """A stored render fingerprint that disagrees with the fingerprint
    recomputed from current authoritative claims surfaces by page_id."""
    ws = Workspace.init(tmp_path)
    store = ClaimStore(ws)
    fps = FingerprintStore(ws)

    initial = EntityContainer(
        entity_id="concept.example",
        entity_type="concept",
        display_name="Example",
        assertions=[
            Assertion(
                claim_id="c.example.1",
                subject_id="concept.example",
                claim_kind="definition",
                claim_text="Original claim text",
                render_targets=[
                    RenderTarget(
                        page_id="concept/example",
                        block_id="claim_block.concept.example",
                    )
                ],
            )
        ],
    )
    store.save_entity(initial)
    fps.save(
        {
            "concept/example": compute_render_fingerprint(
                initial, "claim_block.concept.example"
            )
        }
    )

    mutated = EntityContainer(
        entity_id="concept.example",
        entity_type="concept",
        display_name="Example",
        assertions=[
            Assertion(
                claim_id="c.example.1",
                subject_id="concept.example",
                claim_kind="definition",
                claim_text="MUTATED CLAIM TEXT — fingerprint must now drift",
                render_targets=[
                    RenderTarget(
                        page_id="concept/example",
                        block_id="claim_block.concept.example",
                    )
                ],
            )
        ],
    )
    store.save_entity(mutated)

    report = rebuild(ws, target="health_report")

    assert "concept/example" in report.render_fingerprint_drift


def test_health_report_surfaces_yaml_corrupt_lock_as_held_with_no_owner(
    tmp_path: Path,
) -> None:
    """A YAML-parse-corrupt lock file must not be silently downgraded to
    `lock_held=False`. The report surfaces it as held-with-no-owner so
    the next move is manual `unlock`, not "no lock at all"."""
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    lock.path.parent.mkdir(parents=True, exist_ok=True)
    # Unterminated YAML flow scope — `yaml.safe_load` raises `YAMLError`.
    lock.path.write_text("{not: valid, yaml: [unterminated\n", encoding="utf-8")

    report = rebuild(ws, target="health_report")

    assert report.lock_held is True
    assert report.lock_owner is None
    # The function must still return normally; no crash, no remediation.
    assert isinstance(report, HealthReport)


def test_health_report_surfaces_shape_invalid_lock_as_held_with_no_owner(
    tmp_path: Path,
) -> None:
    """A YAML-valid lock mapping that is missing required keys must
    also surface as held-with-no-owner. The shape-invalid path is
    distinct from the YAML-parse-corrupt path but produces the same
    drift signal."""
    ws = Workspace.init(tmp_path)
    lock = WorkspaceLock(ws)
    lock.path.parent.mkdir(parents=True, exist_ok=True)
    # YAML-valid mapping that does not match the `Lock` shape (missing
    # `lock_id`, `scope`, `op_id`, `owner_id`, `acquired_at`,
    # `heartbeat_at`, `timeout_seconds`).
    lock.path.write_text(
        yaml.safe_dump({"unexpected": "shape"}, sort_keys=True),
        encoding="utf-8",
    )

    report = rebuild(ws, target="health_report")

    assert report.lock_held is True
    assert report.lock_owner is None
    assert isinstance(report, HealthReport)


def test_health_report_emits_zero_drift_on_clean_workspace(tmp_path: Path) -> None:
    """A freshly-initialised workspace has no drift: empty drift lists
    and false drift flags. Sidecars are missing (which is honest, not a
    drift signal in itself)."""
    ws = Workspace.init(tmp_path)

    report = rebuild(ws, target="health_report")

    assert isinstance(report, HealthReport)
    assert report.target == "health_report"
    assert report.interrupted_journals == []
    assert report.stale_lock_unrecoverable is False
    assert report.structure_report_drift == []
    assert report.missing_structure_reports == []
    assert report.render_fingerprint_drift == []
    assert report.entity_count == 0
    assert report.claim_count == 0
    assert report.pending_proposals == 0
    assert report.stale_claims == 0
    assert report.retracted_claims == 0
    assert report.lock_held is False
    assert report.lock_owner is None
    # The derived YAML is persisted deterministically.
    persisted = ws.state_reports_health / "health_report.yaml"
    assert persisted.is_file()
    payload = yaml.safe_load(persisted.read_text(encoding="utf-8"))
    assert payload["target"] == "health_report"
    assert payload["interrupted_journals"] == []
    assert payload["render_fingerprint_drift"] == []
