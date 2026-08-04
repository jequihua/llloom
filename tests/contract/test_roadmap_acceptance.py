"""Final roadmap acceptance suite (Slice 080).

This file is the release-candidate smoke for the
``feedback/2026-05-22_llloom_development_roadmap_synthesis.md``
roadmap (Slices 068–079). It is not exhaustive — every slice
keeps its own per-slice contract file — but it proves the
post-roadmap surfaces compose end-to-end as one coherent
workflow.

Five cross-slice acceptance tests:

1. ``test_seed_apply_then_doctor_last_op_composes_end_to_end``
   — CLI seed apply → library ``doctor(last_op=True)`` review
   bundle. The mandatory CLI smoke per the slice prompt. Covers
   Slices 075 / 075a / 076 / 079.
2. ``test_read_only_inspection_surfaces_compose_after_update``
   — render dry-run, render list-targets, query (default +
   ``status="all"`` + ``ids_only``), claim card, every list
   verb. Covers Slices 073 / 077 / 077a.
3. ``test_supersede_composes_with_query_card_render_and_doctor``
   — lifecycle supersede + downstream inspection surfaces +
   render re-application + doctor cleanliness. Covers Slices
   071 / 077 / 078 / 079.
4. ``test_doctor_accepted_warnings_flow_on_real_updated_workspace``
   — accepted-warning allowlist separation on a real seeded
   workspace; ``doctor`` does not modify any file other than
   the test's allowlist write. Covers Slice 079.
5. ``test_recovery_diagnostics_compose_with_transaction_staging``
   — stale-recoverable lock + abandoned transaction directory
   surfaced by doctor, cleared by ``reconcile()``. Covers
   Slices 069 / 074 / 079.

Every test runs in ``tmp_path``; ``LLMInvoke.invoke`` is
monkeypatched to raise wherever a model call would be a
correctness failure. Read-only claims are snapshot/diff
guarded via :func:`_snapshot_state`.
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
from llloom.ops import (
    apply_seed_manifest,
    claim_card,
    doctor,
    list_claims,
    list_pages,
    list_render_targets,
    list_sources,
    query,
    reconcile,
    render,
    supersede,
)
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.results import (
    ClaimCard,
    DoctorResult,
    SupersedeResult,
    UpdateReviewBundle,
)
from llloom.state.journal import OperationJournal
from llloom.state.lock import WorkspaceLock
from llloom.workspace.layout import Workspace


# Shared fixtures ------------------------------------------------------


SOURCE_TEXT = """\
# Article

## Methods

Alpha is documented in the source. It anchors the roadmap suite.

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
Commentary alpha — this paragraph must survive every roadmap operation.
<!-- /llloom:commentary -->
"""


PAGE_BETA = """\
---
page_id: concept/beta
page_class: concept
write_policy: mixed
status: rendered
---

<!-- llloom:claim-block id=claim_block.concept.beta -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.beta owner=human -->
Commentary beta.
<!-- /llloom:commentary -->
"""


def _seed_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    (ws.pages / "concepts").mkdir(parents=True, exist_ok=True)
    (ws.pages / "concepts" / "alpha.md").write_text(PAGE_ALPHA, encoding="utf-8")
    (ws.pages / "concepts" / "beta.md").write_text(PAGE_BETA, encoding="utf-8")
    return ws


def _manifest_payload() -> dict:
    return {
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


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _seed_two_validated_claims(ws: Workspace) -> None:
    """Build two validated claims under a single entity so the
    supersede acceptance test can exercise the
    ``(validated, "superseded")`` lifecycle edge without going
    through the seed manifest.
    """
    seeds = [
        SeedClaim(
            entity_id="concept.alpha",
            entity_type="concept",
            display_name="Alpha",
            claim_id="c.alpha.1",
            claim_kind="definition",
            claim_text="Alpha is documented in the source. It anchors the roadmap suite.",
            locator=Locator(
                locator_type="markdown_prose_v1",
                heading_path=["Methods"],
                paragraph_index=1,
                sentence_start=1,
                sentence_end=2,
            ),
            render_target=("concept/alpha", "claim_block.concept.alpha"),
        ),
        SeedClaim(
            entity_id="concept.alpha",
            entity_type="concept",
            display_name="Alpha",
            claim_id="c.alpha.2",
            claim_kind="definition",
            claim_text="Alpha is documented in the source. It anchors the roadmap suite.",
            locator=Locator(
                locator_type="markdown_prose_v1",
                heading_path=["Methods"],
                paragraph_index=1,
                sentence_start=1,
                sentence_end=2,
            ),
            render_target=("concept/alpha", "claim_block.concept.alpha"),
        ),
    ]
    ingest(
        ws,
        ws.raw_sources / "article.md",
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=seeds,
    )
    # Promote both claims to validated by direct YAML mutation. This
    # is the same fixture pattern Slice 078's lifecycle test uses;
    # the acceptance suite is read-only over the lifecycle helpers.
    entity_path = ws.claims_entities / "concept.alpha.yaml"
    data = yaml.safe_load(entity_path.read_text(encoding="utf-8"))
    for assertion in data.get("assertions", []):
        assertion["status"] = "validated"
    entity_path.write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )


def _snapshot_state(ws: Workspace) -> dict[str, object]:
    """Read-only snapshot covering every artifact a doctor /
    inspection call must NOT mutate. Includes the
    accepted-warning file presence so tests that intentionally
    write the allowlist can snapshot before / after the
    intentional write.
    """
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
    state["graph_present"] = ws.graph_db.is_file()
    return state


def _patch_llminvoke_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Slice 075 / 076 / 077 / 079 all require that the post-
    roadmap path never invokes a model. The acceptance suite
    enforces this for every test that exercises seed / query /
    doctor surfaces."""
    monkeypatch.setattr(
        LLMInvoke,
        "invoke",
        lambda self, *a, **kw: (_ for _ in ()).throw(
            AssertionError(
                "LLMInvoke.invoke was called during a roadmap acceptance test"
            )
        ),
    )


# ---------------------------------------------------------------------
# 1. CLI seed apply → library doctor(last_op=True)
# ---------------------------------------------------------------------


def test_seed_apply_then_doctor_last_op_composes_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mandatory CLI smoke from the slice prompt.

    Runs ``llloom seed apply manifest.yaml`` through
    ``llloom.cli.main(...)`` (real CLI dispatcher), then calls
    the library ``doctor(workspace, last_op=True)`` and asserts
    the review bundle stitches the Slice 075 / 075a / 076 / 079
    surfaces together end-to-end.
    """
    _patch_llminvoke_raises(monkeypatch)
    ws = _seed_workspace(tmp_path)
    manifest = _write_manifest(tmp_path / "manifest.yaml", _manifest_payload())

    page_before = (ws.pages / "concepts" / "alpha.md").read_text(encoding="utf-8")

    rc = cli_main(
        ["--root", str(tmp_path), "seed", "apply", str(manifest)]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    seed_payload = json.loads(captured.out)
    assert seed_payload.get("refusal_reason") is None
    assert seed_payload["claims_created"], "expected one CreatedClaim"
    created = seed_payload["claims_created"][0]
    assert created["claim_id"] == "c.alpha.1"
    assert created["entity_id"] == "concept.alpha"
    assert created["verification_status"] == "verified"
    assert seed_payload["report_path"] is not None
    assert seed_payload["report_path"].startswith("state/reports/updates/")
    assert "src.article" in seed_payload["sources_planned"]
    op_id = seed_payload["op_ids"][0]

    # Slice 076 update report exists on disk.
    report_path = ws.state_reports_updates / f"{op_id}.yaml"
    assert report_path.is_file()
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    assert report["provenance"] == {
        "generation_mode": "deterministic_seed",
        "model_provider": None,
        "provider_calls": 0,
        "api_cost_usd": 0,
    }

    # Rendered page changed inside the claim block but commentary survived.
    page_after = (ws.pages / "concepts" / "alpha.md").read_text(encoding="utf-8")
    assert page_after != page_before
    assert "Commentary alpha — this paragraph must survive every roadmap operation." in page_after

    # No transaction directory remains after a successful render.
    txn_root = ws.state_transactions
    assert sorted(txn_root.iterdir()) == [], (
        f"unexpected transaction dirs: {list(txn_root.iterdir())}"
    )

    # Library doctor(last_op=True) returns the matching bundle.
    result = doctor(ws, last_op=True)
    assert isinstance(result, DoctorResult)
    bundle = result.update_review
    assert isinstance(bundle, UpdateReviewBundle)
    assert bundle.op_id == op_id
    assert bundle.op_kind == "seed_apply"
    assert bundle.seed_update_report_path == f"state/reports/updates/{op_id}.yaml"
    assert "src.article" in bundle.source_changes
    assert "c.alpha.1" in bundle.claim_changes
    assert bundle.provenance["generation_mode"] == "deterministic_seed"
    assert bundle.provenance["model_provider"] is None
    for key in ("failures", "warnings", "canary_hits"):
        assert key in bundle.lint_summary
    for key in ("verified", "failed", "mismatches"):
        assert key in bundle.verify_summary
    for key in ("source_count", "claim_count", "lock_held"):
        assert key in bundle.status_summary


# ---------------------------------------------------------------------
# 2. Read-only inspection surfaces compose after the update
# ---------------------------------------------------------------------


def test_read_only_inspection_surfaces_compose_after_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slice 073 + 077 + 077a compose against a real updated
    workspace. Every surface is snapshot/diff guarded.
    """
    _patch_llminvoke_raises(monkeypatch)
    ws = _seed_workspace(tmp_path)
    manifest = _write_manifest(tmp_path / "manifest.yaml", _manifest_payload())
    apply_seed_manifest(ws, manifest)

    before = _snapshot_state(ws)

    # Slice 073 read-only render plan surfaces.
    dry = render(ws, dry_run=True)
    list_t = render(ws, list_targets=True)
    assert dry.dry_run is True and dry.list_targets is False
    assert list_t.list_targets is True and list_t.dry_run is False

    # Slice 077 default query returns the visible claim.
    default = query(ws, question="Alpha")
    default_ids = {c["claim_id"] for c in default.citations}
    assert "c.alpha.1" in default_ids

    # Slice 077a default empty query returns no citations.
    empty = query(ws, question="")
    assert empty.citations == [] and empty.used_claim_ids == []

    # Slice 077a explicit inspection: status="all" surfaces.
    all_visible = query(ws, question="", status="all", max_citations=10)
    assert {c["claim_id"] for c in all_visible.citations} == {"c.alpha.1"}

    # Slice 077a documented ids_only=True alone counts as explicit.
    ids_only = query(ws, question="", ids_only=True, max_citations=10)
    assert ids_only.ids_only is True
    assert ids_only.used_claim_ids == ["c.alpha.1"]

    # Slice 077 list operations.
    summaries = list_claims(ws)
    assert {s.claim_id for s in summaries} == {"c.alpha.1"}
    assert summaries[0].qualified_target == "claim:concept.alpha:c.alpha.1"

    card = claim_card(ws, "claim:concept.alpha:c.alpha.1")
    assert isinstance(card, ClaimCard)
    assert card.entity_id == "concept.alpha"
    assert card.evidence and card.evidence[0].source_id == "src.article"

    sources = list_sources(ws)
    assert {s.source_id for s in sources} == {"src.article"}

    pages = list_pages(ws)
    assert {p.page_id for p in pages} >= {"concept/alpha", "concept/beta", "overview"}

    targets = list_render_targets(ws)
    assert any(t.page_id == "concept/alpha" for t in targets)

    after = _snapshot_state(ws)
    assert before == after, (
        "read-only inspection mutated the workspace: "
        f"diff = "
        f"{ {k: (before[k], after[k]) for k in before if before[k] != after[k]} }"
    )


# ---------------------------------------------------------------------
# 3. Lifecycle supersede composes with query, card, render, doctor
# ---------------------------------------------------------------------


def test_supersede_composes_with_query_card_render_and_doctor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slice 078 + 071 + 077 + 079 compose on a workspace with
    two validated claims.

    Workspace setup goes through the legacy ``ingest(...,
    seed_claims=...)`` fixture (which still routes through the
    default ``NullModel`` harness path for the
    ``claim_extract_and_view_render`` policy — that's fine and
    model-free in practice). The LLMInvoke monkeypatch fires
    AFTER setup so the supersede / query / card / render /
    doctor surfaces remain guarded against any model call.
    """
    ws = _seed_workspace(tmp_path)
    _seed_two_validated_claims(ws)
    _patch_llminvoke_raises(monkeypatch)

    result = supersede(
        ws,
        old="claim:concept.alpha:c.alpha.1",
        by="claim:concept.alpha:c.alpha.2",
    )
    assert isinstance(result, SupersedeResult)
    assert result.refused is False
    assert result.old_to_status == "superseded"

    # Default query excludes the superseded claim.
    default = query(ws, question="Alpha", max_citations=10)
    default_ids = {c["claim_id"] for c in default.citations}
    assert "c.alpha.1" not in default_ids
    assert "c.alpha.2" in default_ids

    # status="superseded" surfaces it.
    super_only = query(
        ws, question="Alpha", status="superseded", max_citations=10
    )
    assert {c["claim_id"] for c in super_only.citations} == {"c.alpha.1"}

    # claim_card on the NEW claim shows the supersedes link.
    new_card = claim_card(ws, "claim:concept.alpha:c.alpha.2")
    assert new_card.supersedes == "claim:concept.alpha:c.alpha.1"

    # Render the target page. The [SUPERSEDED] marker should be
    # visible because superseded claims remain render-visible
    # (only retracted / retracted_by_source / archived are hidden).
    render_result = render(ws)
    page_text = (ws.pages / "concepts" / "alpha.md").read_text(encoding="utf-8")
    assert "[SUPERSEDED]" in page_text

    # Doctor reports no false source/hash/page errors for the
    # lifecycle-only change.
    diag = doctor(ws)
    bad_categories = {"source", "page"}
    bad = [w for w in diag.warnings if w.category in bad_categories]
    assert bad == [], (
        "lifecycle-only supersede should not surface source/page errors: "
        f"{[w.warning_id for w in bad]}"
    )
    # Doctor must remain read-only.
    before = _snapshot_state(ws)
    _ = doctor(ws)
    after = _snapshot_state(ws)
    assert before == after


# ---------------------------------------------------------------------
# 4. Doctor accepted-warning flow on a real updated workspace
# ---------------------------------------------------------------------


def test_doctor_accepted_warnings_flow_on_real_updated_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slice 079: accepted-warning allowlist separates known
    signals (missing search sidecar) from new ones (missing
    graph sidecar) on a workspace that has already been
    updated through the seed apply path.
    """
    _patch_llminvoke_raises(monkeypatch)
    ws = _seed_workspace(tmp_path)
    apply_seed_manifest(
        ws, _write_manifest(tmp_path / "manifest.yaml", _manifest_payload())
    )

    # Pass 1: doctor surfaces missing search + graph sidecars.
    pass1 = doctor(ws)
    initial_ids = {w.warning_id for w in pass1.warnings}
    assert "sidecar:search:missing" in initial_ids
    assert "sidecar:graph:missing" in initial_ids

    # Operator-curated allowlist: accept search sidecar, plus an
    # unmatched entry so stale_acceptances has something to report.
    allowlist = ws.state_reports_health / "accepted_warnings.yaml"
    allowlist.parent.mkdir(parents=True, exist_ok=True)
    allowlist.write_text(
        yaml.safe_dump(
            {
                "version": "accepted_warnings_v1",
                "accepted": [
                    {
                        "warning_id": "sidecar:search:missing",
                        "reason": "Hybrid search deferred to next release.",
                        "accepted_by": "architect",
                        "accepted_at": "2026-05-24T00:00:00Z",
                        "evidence": [
                            "05_governance/reviews/review_endorser_search_sidecar_deferral.md"
                        ],
                    },
                    {
                        "warning_id": "sidecar:nonexistent:missing",
                        "reason": "Stale entry left to prove stale_acceptances surfaces.",
                        "evidence": [
                            "05_governance/reviews/example.md"
                        ],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # Snapshot AFTER the test's intentional allowlist write so any
    # extra writes during the second doctor pass are caught.
    before = _snapshot_state(ws)

    pass2 = doctor(ws)
    accepted_ids = {a.warning.warning_id for a in pass2.accepted_warnings}
    remaining_ids = {w.warning_id for w in pass2.warnings}
    assert "sidecar:search:missing" in accepted_ids
    assert "sidecar:search:missing" not in remaining_ids
    # The unmatched allowlist entry surfaces in stale_acceptances.
    assert "sidecar:nonexistent:missing" in pass2.stale_acceptances
    # Graph sidecar warning still remains — allowlist did not cover it.
    assert "sidecar:graph:missing" in remaining_ids

    after = _snapshot_state(ws)
    assert before == after, (
        "doctor mutated files outside the test's intentional allowlist write: "
        f"{ {k: (before[k], after[k]) for k in before if before[k] != after[k]} }"
    )


# ---------------------------------------------------------------------
# 5. Recovery diagnostics compose with transaction staging
# ---------------------------------------------------------------------


def _backdate_lock_heartbeat(lock: WorkspaceLock, seconds_ago: int = 3600) -> None:
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = yaml.safe_load(lock.path.read_text(encoding="utf-8"))
    payload["heartbeat_at"] = past
    lock.path.write_text(
        yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
    )


def test_recovery_diagnostics_compose_with_transaction_staging(
    tmp_path: Path,
) -> None:
    """Slice 069 / 074 / 079: a stale-recoverable lock + a matching
    abandoned render-transaction directory are both surfaced by
    doctor; ``reconcile()`` clears both; the next doctor run shows
    them gone.
    """
    ws = _seed_workspace(tmp_path)

    # Manufacture a stale-recoverable lock with a matching
    # in-progress journal entry so the Slice 069 predicate
    # (timed-out + in-progress journal) succeeds.
    journal = OperationJournal(ws)
    op_id = journal.new_op_id("render")
    journal.start(
        op_id=op_id,
        op_kind="render",
        lock_id="lock.workspace",
    )
    lock = WorkspaceLock(ws)
    lock.acquire(op_id=op_id, owner_id="t", timeout_seconds=1)
    _backdate_lock_heartbeat(lock)

    # Matching abandoned transaction directory.
    txn = ws.state_transactions / op_id
    txn.mkdir(parents=True)
    (txn / "manifest.yaml").write_text(
        "status: staged\n", encoding="utf-8"
    )

    pass1 = doctor(ws)
    pass1_ids = {w.warning_id for w in pass1.warnings}
    assert f"transaction:abandoned:{op_id}" in pass1_ids
    # The stale-recoverable lock surfaces as either the recoverable
    # or interrupted-journal warning (the predicate has two natural
    # views of the same condition).
    lock_warning_present = any(
        wid in pass1_ids
        for wid in (
            f"lock:stale-recoverable:{op_id}",
            f"lock:interrupted-journal:{op_id}",
        )
    )
    assert lock_warning_present, (
        f"expected a recoverable / interrupted lock warning for {op_id}; "
        f"got {sorted(pass1_ids)}"
    )

    # Run reconcile to clear the stale lock + remove the matching
    # abandoned transaction directory.
    rec = reconcile(ws)
    assert rec.lock_cleared
    # The transaction directory matching the cleared op_id is
    # removed by Slice 074's reconcile extension.
    assert not txn.is_dir()

    pass2 = doctor(ws)
    pass2_ids = {w.warning_id for w in pass2.warnings}
    assert f"transaction:abandoned:{op_id}" not in pass2_ids
    assert f"lock:stale-recoverable:{op_id}" not in pass2_ids
    assert f"lock:interrupted-journal:{op_id}" not in pass2_ids
