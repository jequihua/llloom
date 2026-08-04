"""Contract: seed update reports, provenance, and excerpt equality (Slice 076).

Pins the durable audit-evidence surface that sits on top of the Slice
075 / 075a deterministic seed-manifest apply. Each real, mutating
``llloom seed apply`` writes one YAML report under
``state/reports/updates/<op_id>.yaml`` recording planned + created
claims, manifest / source hashes, before / after workspace counts,
bounded excerpt previews, and explicit provenance fields proving the
path never invoked a model.

The manifest schema also gained an optional ``excerpt_equality:
exact_one_sentence`` field; a mismatch refuses the source batch
atomically with a bounded diagnostic and no workspace mutation.

See ``04_specification/seed_manifest_v1.md`` and
``04_specification/storage_and_state_model.md`` for the contracts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llloom.claims.store import ClaimStore
from llloom.llm.harness import LLMInvoke
from llloom.ops import apply_seed_manifest
from llloom.ops.results import CreatedClaim, SeedManifestResult
from llloom.sources.registry import SourceRegistry
from llloom.state import (
    SEED_UPDATE_REPORT_VERSION,
    SeedExcerptCheck,
    check_seed_excerpt_equality,
)
from llloom.state.fingerprints import FingerprintStore
from llloom.state.journal import OperationJournal
from llloom.workspace.layout import Workspace


# A long sentinel sentence whose body must never appear in the report.
# The selected excerpt only covers the first sentence of paragraph 1.
_SENTINEL = (
    "ZeppelinAlpacaSentinelLongUniqueMarker that nothing else in the "
    "fixture mentions and that must never escape into a report."
)
SOURCE_TEXT = (
    "# Article\n\n"
    "## Methods\n\n"
    "Alpha is documented in the source.\n\n"
    f"Second paragraph carries supporting context including {_SENTINEL}.\n"
)


PAGE_TEMPLATE = """\
---
page_id: concept/seed
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.concept.seed -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.seed owner=human -->

Commentary that must survive the seed apply.

<!-- /llloom:commentary -->
"""


def _seed_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "alpha.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    page_path = ws.pages / "concepts" / "seed.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    return ws


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _good_manifest_payload(*, excerpt_equality: str | None = None) -> dict:
    claim: dict = {
        "entity_id": "concept.alpha",
        "display_name": "Alpha",
        "claim_id": "c.alpha.1",
        "claim_text": "Alpha is documented in the source.",
    }
    if excerpt_equality is not None:
        claim["excerpt_equality"] = excerpt_equality
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
                "page_id": "concept/seed",
                "block_id": "claim_block.concept.seed",
            },
        },
        "sources": [
            {
                "path": "raw/sources/alpha.md",
                "source_id": "src.alpha",
                "claims": [claim],
            }
        ],
    }


def _load_report(ws: Workspace, op_id: str) -> dict:
    path = ws.state_reports_updates / f"{op_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------
# 1. Report parses as YAML; required top-level fields present
# ---------------------------------------------------------------------


def test_seed_apply_writes_report_yaml(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", _good_manifest_payload())

    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded, result.refusal_reason
    assert result.report_path is not None
    assert "state/reports/updates" in result.report_path

    report = _load_report(ws, result.op_ids[0])
    assert report["version"] == SEED_UPDATE_REPORT_VERSION
    assert report["op_id"] == result.op_ids[0]
    assert isinstance(report["created_at"], str) and report["created_at"]
    assert report["manifest"]["path"].endswith("manifest.yaml")
    assert report["manifest"]["sha256"].startswith("sha256:")
    assert len(report["sources"]) == 1
    assert report["sources"][0]["source_id"] == "src.alpha"
    assert report["sources"][0]["sha256"].startswith("sha256:")
    assert isinstance(report["claims_planned"], list) and report["claims_planned"]
    assert isinstance(report["claims_created"], list) and report["claims_created"]
    assert report["provenance"] == {
        "generation_mode": "deterministic_seed",
        "model_provider": None,
        "provider_calls": 0,
        "api_cost_usd": 0,
    }


# ---------------------------------------------------------------------
# 2. Report contains created ids and statuses
# ---------------------------------------------------------------------


def test_report_contains_created_ids_and_statuses(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", _good_manifest_payload())

    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded
    assert len(result.claims_created) == 1
    created = result.claims_created[0]
    assert isinstance(created, CreatedClaim)
    assert created.claim_id == "c.alpha.1"
    assert created.entity_id == "concept.alpha"

    report = _load_report(ws, result.op_ids[0])
    created_entries = report["claims_created"]
    assert created_entries == [
        {
            "claim_id": "c.alpha.1",
            "entity_id": "concept.alpha",
            "status": created.status,
            "verification_status": created.verification_status,
        }
    ]
    assert report["entities_touched"] == ["concept.alpha"]
    assert report["sources"][0]["planned_claim_ids"] == ["c.alpha.1"]
    assert report["sources"][0]["created_claim_ids"] == ["c.alpha.1"]


# ---------------------------------------------------------------------
# 3. No raw source body leak
# ---------------------------------------------------------------------


def test_report_does_not_leak_raw_source_body(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    manifest_path = _write_manifest(
        tmp_path / "manifest.yaml",
        _good_manifest_payload(excerpt_equality="exact_one_sentence"),
    )

    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded, result.refusal_reason

    report_path = ws.state_reports_updates / f"{result.op_ids[0]}.yaml"
    report_text = report_path.read_text(encoding="utf-8")

    # The sentinel sentence lives only in the source paragraph that
    # the manifest does NOT select. It must not appear in the report.
    assert _SENTINEL not in report_text
    # The selected one-sentence excerpt may legitimately appear in the
    # excerpt_checks preview but bounded.
    report = yaml.safe_load(report_text)
    for check in report["excerpt_checks"]:
        assert "excerpt_preview" in check
        assert len(check["excerpt_preview"]) <= 240


# ---------------------------------------------------------------------
# 4. Excerpt equality detects match
# ---------------------------------------------------------------------


def test_excerpt_equality_match_recorded(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    manifest_path = _write_manifest(
        tmp_path / "manifest.yaml",
        _good_manifest_payload(excerpt_equality="exact_one_sentence"),
    )

    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded, result.refusal_reason
    assert len(result.excerpt_checks) == 1
    check = result.excerpt_checks[0]
    assert isinstance(check, SeedExcerptCheck)
    assert check.claim_id == "c.alpha.1"
    assert check.mode == "exact_one_sentence"
    assert check.matched is True
    assert check.message is None
    assert check.excerpt_hash.startswith("sha256:")

    report = _load_report(ws, result.op_ids[0])
    rec = report["excerpt_checks"][0]
    assert rec["mode"] == "exact_one_sentence"
    assert rec["matched"] is True


# ---------------------------------------------------------------------
# 5. Excerpt equality detects mismatch before persistence
# ---------------------------------------------------------------------


def test_excerpt_equality_mismatch_refuses_before_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload(excerpt_equality="exact_one_sentence")
    payload["sources"][0]["claims"][0]["claim_text"] = (
        "Alpha is described differently from the source."
    )
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    page_before = (ws.pages / "concepts" / "seed.md").read_text(encoding="utf-8")
    fps_before = FingerprintStore(ws).load()

    def _refuse_invoke(self, *args, **kwargs):  # noqa: ARG001
        raise AssertionError("LLMInvoke.invoke was called during seed apply")

    monkeypatch.setattr(LLMInvoke, "invoke", _refuse_invoke)

    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded is False
    assert result.refusal_reason is not None
    assert "excerpt_equality" in result.refusal_reason
    # Diagnostic is bounded — no raw source body leak.
    assert _SENTINEL not in result.refusal_reason

    # Zero workspace mutation.
    assert SourceRegistry(ws).list_ids() == []
    assert not ClaimStore(ws).exists("concept.alpha")
    assert (ws.pages / "concepts" / "seed.md").read_text(encoding="utf-8") == page_before
    assert FingerprintStore(ws).load() == fps_before
    assert not (ws.state_locks / "workspace.yaml").is_file()
    # No report written.
    assert list(ws.state_reports_updates.glob("*.yaml")) == []
    # No seed_apply journal entry.
    journal_files = list(ws.state_journals.glob("*.yaml"))
    seed_apply_entries = []
    for jf in journal_files:
        payload = yaml.safe_load(jf.read_text(encoding="utf-8")) or {}
        if payload.get("op_kind") == "seed_apply":
            seed_apply_entries.append(jf)
    assert seed_apply_entries == []


# ---------------------------------------------------------------------
# 6. Provenance: no provider calls
# ---------------------------------------------------------------------


def test_report_provenance_no_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _seed_workspace(tmp_path)
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", _good_manifest_payload())

    def _refuse_invoke(self, *args, **kwargs):  # noqa: ARG001
        raise AssertionError(
            "LLMInvoke.invoke was called during deterministic seed apply"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _refuse_invoke)

    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded, result.refusal_reason

    report = _load_report(ws, result.op_ids[0])
    prov = report["provenance"]
    assert prov["generation_mode"] == "deterministic_seed"
    assert prov["model_provider"] is None
    assert prov["provider_calls"] == 0
    assert prov["api_cost_usd"] == 0

    # Journal entry still has empty invocation_logs.
    journal = OperationJournal(ws)
    entry = journal.load(result.op_ids[0])
    assert entry.op_kind == "seed_apply"
    assert entry.invocation_logs == []


# ---------------------------------------------------------------------
# 7. Dry-run writes no report
# ---------------------------------------------------------------------


def test_dry_run_writes_no_report(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    manifest_path = _write_manifest(
        tmp_path / "manifest.yaml",
        _good_manifest_payload(excerpt_equality="exact_one_sentence"),
    )

    # Capture state-before snapshots; everything must be byte-identical
    # after the dry-run.
    page_before = (ws.pages / "concepts" / "seed.md").read_text(encoding="utf-8")
    fps_before = FingerprintStore(ws).load()
    sources_before = SourceRegistry(ws).list_ids()

    result = apply_seed_manifest(ws, manifest_path, dry_run=True)
    assert result.succeeded, result.refusal_reason
    assert result.dry_run is True
    assert result.report_path is None
    assert result.claims_created == []
    # Dry-run still ran the equality check and recorded the result.
    assert len(result.excerpt_checks) == 1
    assert result.excerpt_checks[0].matched is True

    # No report directory contents.
    assert list(ws.state_reports_updates.glob("*.yaml")) == []
    # No workspace mutation.
    assert SourceRegistry(ws).list_ids() == sources_before
    assert not ClaimStore(ws).exists("concept.alpha")
    assert (ws.pages / "concepts" / "seed.md").read_text(encoding="utf-8") == page_before
    assert FingerprintStore(ws).load() == fps_before
    assert not (ws.state_locks / "workspace.yaml").is_file()


# ---------------------------------------------------------------------
# 8. Before/after counts are useful
# ---------------------------------------------------------------------


def test_before_after_counts_are_useful(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", _good_manifest_payload())

    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded, result.refusal_reason

    assert result.counts_before["sources"] == 0
    assert result.counts_before["entities"] == 0
    assert result.counts_before["claims"] == 0

    assert result.counts_after["sources"] == 1
    assert result.counts_after["entities"] == 1
    assert result.counts_after["claims"] == 1
    # Page count is stable: the apply re-renders an existing page;
    # it does not create new page files.
    assert result.counts_after["pages"] == result.counts_before["pages"]

    report = _load_report(ws, result.op_ids[0])
    assert report["counts"]["before"] == result.counts_before
    assert report["counts"]["after"] == result.counts_after


# ---------------------------------------------------------------------
# 9. Pure helper: check_seed_excerpt_equality on its own
# ---------------------------------------------------------------------


def test_check_seed_excerpt_equality_pure_helper() -> None:
    """The helper is pure (no I/O) and should reuse the verifier's
    whitespace-collapse rule so trivial differences such as line
    wrapping or trailing whitespace do not cause spurious refusals.
    """
    # Identical aside from whitespace runs.
    check = check_seed_excerpt_equality(
        claim_id="c.x",
        claim_text="The first sentence is here.",
        resolved_excerpt="The   first\n  sentence\tis here.  ",
        mode="exact_one_sentence",
    )
    assert check.matched is True
    assert check.message is None

    # Genuine textual difference.
    mismatch = check_seed_excerpt_equality(
        claim_id="c.y",
        claim_text="The first sentence is here.",
        resolved_excerpt="A different sentence.",
        mode="exact_one_sentence",
    )
    assert mismatch.matched is False
    assert mismatch.message is not None
    assert "exact_one_sentence" in mismatch.message

    # Mode "none" never refuses.
    skipped = check_seed_excerpt_equality(
        claim_id="c.z",
        claim_text="anything",
        resolved_excerpt="something else entirely",
        mode="none",
    )
    assert skipped.matched is True
    assert skipped.mode == "none"
