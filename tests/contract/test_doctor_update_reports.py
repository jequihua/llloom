"""Contract: doctor, update reports, and accepted warnings (Slice 079).

Pins the read-only diagnostic surface ``llloom doctor`` adds on top
of the Slice 053 / 075 / 075a / 076 / 077 / 078 substrate. Every
operation is **read-only** — no workspace lock acquisition, no
journal entry, no sidecar / page / fingerprint / claim / source /
report / transaction write, no model / provider call.

Test surface:

1. doctor is read-only (snapshot/diff + LLMInvoke monkeypatch);
2. lock / journal / transaction drift detectors fire;
3. render / sidecar / structure-report drift detectors fire;
4. lifecycle / source / page anomaly detectors fire;
5. accepted-warning allowlist separates known signals from new ones;
6. update review bundle works by op id and last op;
7. CLI behavior + verb-count guard at 25.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from llloom.claims.models import Locator
from llloom.claims.store import ClaimStore
from llloom.cli import main as cli_main
from llloom.llm.harness import LLMInvoke
from llloom.ops import doctor
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.results import (
    AcceptedDoctorWarning,
    DoctorResult,
    DoctorWarning,
    UpdateReviewBundle,
)
from llloom.sources.registry import SourceRegistry
from llloom.state.fingerprints import FingerprintStore
from llloom.state.journal import OperationJournal
from llloom.state.lock import WorkspaceLock
from llloom.workspace.layout import Workspace


SOURCE_TEXT = """\
# Article

## Methods

Alpha is documented in the source. It anchors the doctor slice.

Beta is a separate sentence about a different entity.
"""


PAGE_ALPHA = """\
---
page_id: concept/alpha
page_class: concept
write_policy: mixed
status: rendered
---

<!-- llloom:claim-block id=claim_block.concept.alpha -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.alpha owner=human -->
Commentary alpha.
<!-- /llloom:commentary -->
"""


def _seed_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    (ws.pages / "concepts").mkdir(parents=True, exist_ok=True)
    (ws.pages / "concepts" / "alpha.md").write_text(PAGE_ALPHA, encoding="utf-8")
    return ws


def _seed_claim(ws: Workspace) -> None:
    seed = SeedClaim(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        claim_id="c.alpha.1",
        claim_kind="definition",
        claim_text="Alpha is documented in the source. It anchors the doctor slice.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=2,
        ),
        render_target=("concept/alpha", "claim_block.concept.alpha"),
    )
    ingest(
        ws,
        ws.raw_sources / "article.md",
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[seed],
    )


def _snapshot_state(ws: Workspace) -> dict[str, object]:
    state: dict[str, object] = {}
    state["claims"] = {
        p.name: p.read_bytes() for p in sorted(ws.claims_entities.glob("*.yaml"))
    }
    state["sources"] = {
        p.name: p.read_bytes()
        for p in sorted(ws.state_source_registry.glob("*.yaml"))
    }
    state["pages"] = {
        str(p.relative_to(ws.pages)): p.read_bytes()
        for p in sorted(ws.pages.rglob("*.md"))
    }
    state["fingerprints"] = (
        ws.render_fingerprints.read_bytes()
        if ws.render_fingerprints.is_file()
        else None
    )
    state["lock_present"] = (ws.state_locks / "workspace.yaml").is_file()
    state["journals"] = sorted(p.name for p in ws.state_journals.glob("*.yaml"))
    state["transactions"] = sorted(
        str(p.relative_to(ws.state_transactions))
        for p in ws.state_transactions.rglob("*")
    )
    state["seed_reports"] = sorted(
        p.name for p in ws.state_reports_updates.glob("*.yaml")
    )
    state["health_reports"] = sorted(
        p.name for p in ws.state_reports_health.glob("*.yaml")
    )
    state["search_present"] = ws.search_db.is_file()
    return state


def _backdate_lock_heartbeat(lock: WorkspaceLock, seconds_ago: int = 3600) -> None:
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = yaml.safe_load(lock.path.read_text(encoding="utf-8"))
    payload["heartbeat_at"] = past
    lock.path.write_text(
        yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
    )


def _wid(result: DoctorResult, warning_id: str) -> DoctorWarning | None:
    for w in result.warnings:
        if w.warning_id == warning_id:
            return w
    return None


# ---------------------------------------------------------------------
# 1. Doctor is read-only.
# ---------------------------------------------------------------------


def test_doctor_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claim(ws)

    monkeypatch.setattr(
        LLMInvoke,
        "invoke",
        lambda self, *a, **kw: (_ for _ in ()).throw(
            AssertionError("LLMInvoke.invoke was called from doctor")
        ),
    )

    before = _snapshot_state(ws)
    result = doctor(ws)
    assert isinstance(result, DoctorResult)
    after = _snapshot_state(ws)
    assert before == after, (
        "doctor mutated the workspace: "
        f"diff = "
        f"{ {k: (before[k], after[k]) for k in before if before[k] != after[k]} }"
    )


# ---------------------------------------------------------------------
# 2. Lock / journal / transaction detectors fire.
# ---------------------------------------------------------------------


def test_doctor_detects_lock_and_transaction_problems(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claim(ws)

    # Stale unrecoverable lock: timed-out + no matching in-progress
    # journal entry.
    lock = WorkspaceLock(ws)
    lock.acquire(
        op_id="op.ingest.20990101T000000Z",
        owner_id="t",
        timeout_seconds=1,
    )
    _backdate_lock_heartbeat(lock)

    # Abandoned render transaction directory.
    txn_dir = ws.state_transactions / "op.render.abandoned.0001"
    txn_dir.mkdir(parents=True)
    (txn_dir / "manifest.yaml").write_text(
        "status: staged\n", encoding="utf-8"
    )

    # Interrupted journal entry whose op_id does not match the lock.
    journal = OperationJournal(ws)
    abandoned_op_id = journal.new_op_id("ingest")
    journal.start(
        op_id=abandoned_op_id,
        op_kind="ingest",
        lock_id="lock.workspace",
    )

    result = doctor(ws)

    assert (
        _wid(result, "lock:stale-unrecoverable:op.ingest.20990101T000000Z")
        is not None
    )
    interrupted = _wid(result, f"lock:interrupted-journal:{abandoned_op_id}")
    assert interrupted is not None
    assert interrupted.recommended_command == "llloom reconcile"
    txn = _wid(result, "transaction:abandoned:op.render.abandoned.0001")
    assert txn is not None
    assert txn.recommended_command == "llloom reconcile"
    # recommended_next_commands aggregates de-duplicates.
    assert "llloom reconcile" in result.recommended_next_commands


# ---------------------------------------------------------------------
# 3. Render / sidecar / structure-report drift detectors fire.
# ---------------------------------------------------------------------


def test_doctor_detects_render_and_sidecar_drift(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claim(ws)

    # Corrupt the render fingerprint for the rendered page.
    fps = FingerprintStore(ws)
    current = fps.load()
    assert "concept/alpha" in current
    current["concept/alpha"] = "sha256:" + "0" * 64
    fps.save(current)

    result = doctor(ws)

    drift = _wid(result, "render:fingerprint-drift:concept/alpha")
    assert drift is not None
    assert drift.recommended_command == "llloom reconcile"

    # Search and graph sidecars are missing on a fresh workspace.
    search_w = _wid(result, "sidecar:search:missing")
    assert search_w is not None
    assert search_w.recommended_command == "llloom rebuild search"
    graph_w = _wid(result, "sidecar:graph:missing")
    assert graph_w is not None
    assert graph_w.recommended_command == "llloom rebuild graph"


def test_doctor_detects_missing_structure_report(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    # Register a structured_yaml source so the resolved policy is
    # ``structure_extract``; never run the extraction so the
    # structure report is absent.
    raw_path = ws.raw_sources / "policies.yaml"
    raw_path.write_text(
        "policies:\n  markdown_prose: claim_extract\n", encoding="utf-8"
    )
    registry = SourceRegistry(ws)
    record, _state = registry.register(
        source_id="src.policies",
        raw_path=raw_path,
        source_class="structured_yaml",
    )

    result = doctor(ws)

    missing = _wid(result, f"structure-report:missing:{record.source_id}")
    assert missing is not None
    assert missing.severity == "warning"


# ---------------------------------------------------------------------
# 4. Lifecycle / source / page anomaly detectors fire.
# ---------------------------------------------------------------------


def test_doctor_detects_lifecycle_and_source_and_page_anomalies(
    tmp_path: Path,
) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claim(ws)

    # Flip the claim to ``stale`` directly so the lifecycle detector
    # has something to surface.
    entity_path = ws.claims_entities / "concept.alpha.yaml"
    data = yaml.safe_load(entity_path.read_text(encoding="utf-8"))
    data["assertions"][0]["status"] = "stale"
    entity_path.write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )

    # Source hash drift: change the on-disk source bytes after
    # registration.
    (ws.raw_sources / "article.md").write_text(
        SOURCE_TEXT + "\n## New section\n\nDrifted content.\n",
        encoding="utf-8",
    )

    # Malformed page (non-spine): break the variant-(B) markers.
    bad_page = ws.pages / "concepts" / "broken.md"
    bad_page.write_text(
        "---\npage_id: concept/broken\npage_class: concept\n"
        "write_policy: mixed\nstatus: draft\n---\n\nno markers here.\n",
        encoding="utf-8",
    )

    result = doctor(ws)

    stale = _wid(result, "lifecycle:stale-claims")
    assert stale is not None
    assert stale.severity == "warning"

    drift = _wid(result, "source:hash-drift:src.article")
    assert drift is not None
    assert drift.severity == "error"

    page_w = _wid(result, "page:marker-parse-error:pages/concepts/broken.md")
    assert page_w is not None


def test_doctor_skips_spine_pages_when_checking_markers(tmp_path: Path) -> None:
    """The Overview page (and any spine glob) is exempt from the
    variant-(B) marker contract. The doctor must NOT emit a
    ``page:marker-parse-error`` warning for the starter Overview page.
    """
    ws = _seed_workspace(tmp_path)
    # The starter Overview lives at pages/overview.md and has no
    # claim-block markers.
    assert (ws.pages / "overview.md").is_file()

    result = doctor(ws)

    spine_warnings = [
        w for w in result.warnings
        if w.warning_id == "page:marker-parse-error:pages/overview.md"
    ]
    assert spine_warnings == [], (
        f"spine page should not emit marker warnings: {spine_warnings}"
    )


# ---------------------------------------------------------------------
# 5. Accepted-warning allowlist separates known signals.
# ---------------------------------------------------------------------


def test_doctor_accepted_warnings_are_separated(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claim(ws)

    accepted = ws.state_reports_health / "accepted_warnings.yaml"
    accepted.parent.mkdir(parents=True, exist_ok=True)
    accepted.write_text(
        yaml.safe_dump(
            {
                "version": "accepted_warnings_v1",
                "accepted": [
                    {
                        "warning_id": "sidecar:search:missing",
                        "reason": "Sidecar build deferred to next release.",
                        "accepted_by": "architect",
                        "accepted_at": "2026-05-23T00:00:00Z",
                        "evidence": ["05_governance/reviews/example.md"],
                    },
                    {
                        "warning_id": "sidecar:nonexistent:missing",
                        "reason": "Stale entry — sidecar is unused.",
                        "evidence": ["05_governance/reviews/example.md"],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = doctor(ws)

    accepted_ids = {a.warning.warning_id for a in result.accepted_warnings}
    warning_ids = {w.warning_id for w in result.warnings}
    assert "sidecar:search:missing" in accepted_ids
    assert "sidecar:search:missing" not in warning_ids
    # Graph sidecar still surfaces because the allowlist did not cover it.
    assert "sidecar:graph:missing" in warning_ids
    # Stale acceptance lists the unused entry.
    assert "sidecar:nonexistent:missing" in result.stale_acceptances


def test_doctor_malformed_accepted_warnings_surface_warning(tmp_path: Path) -> None:
    """A missing reason / missing evidence allowlist entry must not
    accept the warning and must emit an
    `accepted-warnings:malformed-entry` diagnostic.
    """
    ws = _seed_workspace(tmp_path)
    accepted = ws.state_reports_health / "accepted_warnings.yaml"
    accepted.parent.mkdir(parents=True, exist_ok=True)
    accepted.write_text(
        yaml.safe_dump(
            {
                "version": "accepted_warnings_v1",
                "accepted": [
                    {
                        "warning_id": "sidecar:search:missing",
                        # reason omitted — malformed
                        "evidence": ["05_governance/reviews/example.md"],
                    },
                    {
                        "warning_id": "sidecar:graph:missing",
                        "reason": "Graph sidecar deferred.",
                        # evidence omitted — malformed
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = doctor(ws)

    warning_ids = {w.warning_id for w in result.warnings}
    accepted_ids = {a.warning.warning_id for a in result.accepted_warnings}
    # Neither malformed entry was accepted.
    assert "sidecar:search:missing" in warning_ids
    assert "sidecar:search:missing" not in accepted_ids
    assert "sidecar:graph:missing" in warning_ids
    assert "sidecar:graph:missing" not in accepted_ids
    # A dedicated malformed-entry warning was raised.
    malformed = [
        w for w in result.warnings if w.category == "accepted-warnings"
    ]
    assert malformed and "malformed" in malformed[0].warning_id


# ---------------------------------------------------------------------
# 6. Update review bundle by op id and last op.
# ---------------------------------------------------------------------


def test_doctor_update_review_bundle_by_op_id_and_last_op(tmp_path: Path) -> None:
    """The seed-apply path writes a Slice 076 update report; the
    doctor bundle should expose its source / claim / rendered-page
    fields plus the provenance + lint / verify / status summaries.
    """
    from llloom.ops import apply_seed_manifest

    ws = _seed_workspace(tmp_path)
    # Seed apply through a tiny manifest so a journal entry + update
    # report exist for the doctor bundle.
    manifest = {
        "version": "seed_manifest_v1",
        "defaults": {
            "source_class": "markdown_prose",
            "entity_type": "concept",
            "claim_kind": "definition",
            "status": "draft",
            "locator": {
                "locator_type": "markdown_prose_v1",
                "heading_path": ["Methods"],
                "paragraph_index": 1,
                "sentence_start": 1,
                "sentence_end": 1,
            },
            "render_target": {
                "page_id": "concept/alpha",
                "block_id": "claim_block.concept.alpha",
            },
        },
        "sources": [
            {
                "path": "raw/sources/article.md",
                "source_id": "src.article",
                "claims": [
                    {
                        "entity_id": "concept.alpha",
                        "display_name": "Alpha",
                        "claim_id": "c.alpha.1",
                        "claim_text": "Alpha is documented in the source.",
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    seed_result = apply_seed_manifest(ws, manifest_path)
    assert seed_result.succeeded, seed_result.refusal_reason
    op_id = seed_result.op_ids[0]

    # By op id.
    by_id = doctor(ws, op_id=op_id)
    bundle = by_id.update_review
    assert isinstance(bundle, UpdateReviewBundle)
    assert bundle.op_id == op_id
    assert bundle.op_kind == "seed_apply"
    assert bundle.journal_status in {"completed", "in_progress"}
    assert bundle.seed_update_report_path is not None
    assert bundle.seed_update_report_path.startswith("state/reports/updates/")
    assert "src.article" in bundle.source_changes
    assert "c.alpha.1" in bundle.claim_changes
    assert bundle.provenance.get("generation_mode") == "deterministic_seed"
    assert bundle.provenance.get("model_provider") is None
    for required_key in ("failures", "warnings", "canary_hits"):
        assert required_key in bundle.lint_summary
    for required_key in ("verified", "failed", "mismatches"):
        assert required_key in bundle.verify_summary
    for required_key in ("source_count", "claim_count", "lock_held"):
        assert required_key in bundle.status_summary

    # By last op.
    by_last = doctor(ws, last_op=True)
    assert by_last.update_review is not None
    assert by_last.update_review.op_id == op_id


def test_doctor_review_bundle_for_non_seed_op_has_no_seed_report(
    tmp_path: Path,
) -> None:
    """A non-seed operation (the starter `rebuild health_report`
    journal entry, for example) still produces a journal-backed
    bundle but `seed_update_report_path` stays None.
    """
    from llloom.ops import rebuild

    ws = _seed_workspace(tmp_path)
    rebuild(ws, target="health_report")
    journal = OperationJournal(ws)
    last = journal.latest()
    assert last is not None

    result = doctor(ws, op_id=last.op_id)
    bundle = result.update_review
    assert bundle is not None
    assert bundle.op_id == last.op_id
    assert bundle.seed_update_report_path is None
    assert bundle.source_changes == []
    assert bundle.claim_changes == []


def test_doctor_unknown_op_id_returns_review_bundle_warning(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    result = doctor(ws, op_id="op.does.not.exist")
    assert result.update_review is None
    review_warnings = [
        w for w in result.warnings if w.category == "review-bundle"
    ]
    assert review_warnings
    assert "unknown-op-id" in review_warnings[0].warning_id


# ---------------------------------------------------------------------
# 7. CLI behavior and verb count.
# ---------------------------------------------------------------------


def test_cli_doctor_prints_json_and_handles_flag_combos(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claim(ws)

    # Plain doctor: prints JSON. A fresh workspace has missing
    # sidecars (info severity) but no errors → exit 0.
    rc = cli_main(["--root", str(tmp_path), "doctor"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["target"] == "doctor"
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["accepted_warnings"], list)
    assert "recommended_next_commands" in payload

    # --op-id and --last-op together is refused cleanly.
    rc_bad = cli_main(
        [
            "--root",
            str(tmp_path),
            "doctor",
            "--op-id",
            "op.foo",
            "--last-op",
        ]
    )
    captured_bad = capsys.readouterr()
    assert rc_bad == 1
    assert "at most one" in captured_bad.err
