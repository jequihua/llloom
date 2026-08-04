"""Contract: seed manifest CLI v1 (Slice 075).

Pins the deterministic, model-free seed-manifest application
surface. ``llloom seed apply <manifest.yaml>`` and the library
entry point :func:`llloom.ops.apply_seed_manifest` route through
the existing batch-atomic verifier + claim-store primitives the
``ingest`` path uses for deterministic seed claims; no
``LLMInvoke`` / ``NullModel`` / model provider is touched.

See ``04_specification/seed_manifest_v1.md`` for the schema and
merge contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llloom.claims.store import ClaimStore
from llloom.cli import main as cli_main
from llloom.llm.harness import LLMInvoke
from llloom.ops import apply_seed_manifest
from llloom.ops.results import CreatedClaim, PlannedSeedClaim
from llloom.sources.registry import SourceRegistry, SourceRegistryError
from llloom.state.fingerprints import FingerprintStore
from llloom.state.journal import OperationJournal
from llloom.workspace.layout import Workspace


SOURCE_TEXT = """\
# Article

## Methods

Alpha is documented in the source. It anchors the seed-manifest contract.

The second paragraph contains supporting context for downstream claims.
"""


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


def _good_manifest_payload() -> dict:
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
                "claims": [
                    {
                        "entity_id": "concept.alpha",
                        "display_name": "Alpha",
                        "claim_id": "c.alpha.1",
                        "claim_text": (
                            "Alpha is documented in the source. It anchors "
                            "the seed-manifest contract."
                        ),
                    }
                ],
            }
        ],
    }


def _journal_count(ws: Workspace, op_kind: str | None = None) -> int:
    if not ws.state_journals.is_dir():
        return 0
    total = 0
    for path in ws.state_journals.glob("*.yaml"):
        if op_kind is None:
            total += 1
        else:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if payload.get("op_kind") == op_kind:
                total += 1
    return total


# ---------------------------------------------------------------------
# 1. Manifest defaults merge correctly
# ---------------------------------------------------------------------


def test_manifest_defaults_merge_with_per_claim_status_winning_over_cli_status(
    tmp_path: Path,
) -> None:
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload()
    # Add a second claim that explicitly carries status="validated"; the
    # first one (no status) should inherit the manifest default "draft"
    # unless --status promotes it.
    payload["sources"][0]["claims"].append(
        {
            "entity_id": "concept.alpha",
            "display_name": "Alpha",
            "claim_id": "c.alpha.2",
            "claim_text": (
                "The second paragraph contains supporting context for "
                "downstream claims."
            ),
            "status": "validated",
            "locator": {
                "paragraph_index": 2,
                "sentence_start": 1,
                "sentence_end": 1,
            },
        }
    )
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    # CLI --status="reviewed" should fill c.alpha.1 (no per-claim status)
    # but must NOT override c.alpha.2's explicit "validated".
    result = apply_seed_manifest(ws, manifest_path, status="reviewed")
    assert result.succeeded, result.refusal_reason
    statuses = {p.claim_id: p.status for p in result.claims_planned}
    # c.alpha.1 has manifest default "draft", so CLI --status only fires
    # when the merged claim has no status. Since manifest defaults set
    # status="draft", c.alpha.1 retains "draft".
    assert statuses["c.alpha.1"] == "draft"
    assert statuses["c.alpha.2"] == "validated"

    # Persisted assertions reflect the merged statuses.
    entity = ClaimStore(ws).load_entity("concept.alpha")
    assertion_1 = entity.find_assertion("c.alpha.1")
    assertion_2 = entity.find_assertion("c.alpha.2")
    assert assertion_1 is not None and assertion_1.status == "draft"
    assert assertion_2 is not None and assertion_2.status == "validated"


def test_cli_status_fills_when_manifest_omits_status(tmp_path: Path) -> None:
    """When the manifest omits status everywhere, CLI --status fills."""
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload()
    payload["defaults"].pop("status")  # no manifest-level default
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    result = apply_seed_manifest(ws, manifest_path, status="reviewed")
    assert result.succeeded, result.refusal_reason
    assert result.claims_planned[0].status == "reviewed"
    entity = ClaimStore(ws).load_entity("concept.alpha")
    assertion = entity.find_assertion("c.alpha.1")
    assert assertion is not None and assertion.status == "reviewed"


# ---------------------------------------------------------------------
# 2. Invalid manifest refuses before persistence
# ---------------------------------------------------------------------


def test_invalid_manifest_refuses_before_any_persistence(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    journals_before = _journal_count(ws)
    locks_before = (ws.state_locks / "workspace.yaml").is_file()
    payload = _good_manifest_payload()
    # Sources must be a non-empty list; replace with an empty list.
    payload["sources"] = []
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded is False
    assert result.refusal_reason
    assert "sources" in result.refusal_reason

    # Zero workspace mutation: no source registered, no claim YAML, no
    # rendered page, no fingerprint write, no journal entry, no lock.
    registry = SourceRegistry(ws)
    assert registry.list_ids() == []
    store = ClaimStore(ws)
    assert not store.exists("concept.alpha")
    assert _journal_count(ws) == journals_before
    assert (ws.state_locks / "workspace.yaml").is_file() == locks_before


# ---------------------------------------------------------------------
# 3. Invalid locator refuses batch atomically (per source entry)
# ---------------------------------------------------------------------


def test_invalid_locator_refuses_source_batch_atomically(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload()
    # Two claims; the second one targets a non-existent paragraph.
    payload["sources"][0]["claims"].append(
        {
            "entity_id": "concept.alpha",
            "display_name": "Alpha",
            "claim_id": "c.alpha.bad",
            "claim_text": "Nonexistent claim that the locator cannot resolve.",
            "locator": {
                "paragraph_index": 999,
                "sentence_start": 1,
                "sentence_end": 1,
            },
        }
    )
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded is False
    assert result.refusal_reason
    assert "verifier refused" in result.refusal_reason
    assert "c.alpha.bad" in result.refusal_reason or "paragraph_index" in result.refusal_reason.lower()

    # Batch-atomic: neither the valid nor the invalid claim landed.
    store = ClaimStore(ws)
    assert not store.exists("concept.alpha"), (
        "verifier refusal must roll back the whole source-entry batch"
    )
    # No render fingerprint write either (the page was never rendered).
    fps = FingerprintStore(ws).load()
    assert "concept/seed" not in fps


# ---------------------------------------------------------------------
# 4. Dry run writes nothing
# ---------------------------------------------------------------------


def test_dry_run_writes_nothing_but_reports_planned_claims(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload()
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    # Snapshot the workspace before.
    page_text_before = (ws.pages / "concepts" / "seed.md").read_text(encoding="utf-8")
    journals_before = _journal_count(ws)
    sources_before = SourceRegistry(ws).list_ids()
    fps_before = FingerprintStore(ws).load()

    result = apply_seed_manifest(ws, manifest_path, dry_run=True)

    assert result.succeeded, result.refusal_reason
    assert result.dry_run is True
    # Planned claims surface even on dry-run.
    assert len(result.claims_planned) == 1
    planned = result.claims_planned[0]
    assert isinstance(planned, PlannedSeedClaim)
    assert planned.claim_id == "c.alpha.1"
    assert planned.entity_id == "concept.alpha"
    assert planned.status == "draft"
    # claims_created stays empty on dry-run.
    assert result.claims_created == []
    assert result.entities_touched == []
    assert result.pages_rendered == []
    assert result.op_ids == []

    # Zero workspace mutation.
    assert (ws.pages / "concepts" / "seed.md").read_text(encoding="utf-8") == page_text_before
    assert _journal_count(ws) == journals_before
    assert SourceRegistry(ws).list_ids() == sources_before
    assert FingerprintStore(ws).load() == fps_before
    assert not ClaimStore(ws).exists("concept.alpha")
    assert not (ws.state_locks / "workspace.yaml").is_file()


# ---------------------------------------------------------------------
# 5. Created result is honest
# ---------------------------------------------------------------------


def test_real_apply_emits_structured_created_claims(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload()
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    result = apply_seed_manifest(ws, manifest_path, status="reviewed")
    assert result.succeeded, result.refusal_reason
    assert len(result.claims_created) == 1
    created = result.claims_created[0]
    assert isinstance(created, CreatedClaim)
    assert created.claim_id == "c.alpha.1"
    assert created.entity_id == "concept.alpha"
    # CLI --status="reviewed" doesn't override the manifest default
    # status="draft"; per-claim default wins.
    assert created.status == "draft"
    assert created.verification_status == "verified"
    assert result.entities_touched == ["concept.alpha"]
    assert any(p.endswith("seed.md") for p in result.pages_rendered)
    assert result.render_skipped is False


# ---------------------------------------------------------------------
# 6. No model or provider invocation
# ---------------------------------------------------------------------


def test_apply_seed_manifest_never_invokes_llminvoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload()
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    def _refuse_invoke(self, *args, **kwargs):  # noqa: ARG001
        raise AssertionError(
            "LLMInvoke.invoke was called during deterministic seed apply"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _refuse_invoke)

    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded, result.refusal_reason

    # The op_kind="seed_apply" journal entry exists with empty
    # invocation_logs and the deterministic-no-provider notes.
    assert len(result.op_ids) == 1
    journal = OperationJournal(ws)
    entry = journal.load(result.op_ids[0])
    assert entry.op_kind == "seed_apply"
    assert entry.invocation_logs == []
    notes_blob = " ".join(entry.notes)
    assert "deterministic seed manifest" in notes_blob
    assert "provider: none" in notes_blob


# ---------------------------------------------------------------------
# 7. No-render mode
# ---------------------------------------------------------------------


def test_no_render_persists_claims_and_skips_rendering(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload()
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    page_text_before = (ws.pages / "concepts" / "seed.md").read_text(encoding="utf-8")
    fps_before = FingerprintStore(ws).load()

    result = apply_seed_manifest(ws, manifest_path, no_render=True)
    assert result.succeeded, result.refusal_reason
    assert result.render_skipped is True
    assert result.pages_rendered == []

    # Claim persisted.
    entity = ClaimStore(ws).load_entity("concept.alpha")
    assert entity.find_assertion("c.alpha.1") is not None

    # Page and fingerprint store unchanged.
    assert (ws.pages / "concepts" / "seed.md").read_text(encoding="utf-8") == page_text_before
    assert FingerprintStore(ws).load() == fps_before

    # Journal entry has the explicit no-render note.
    journal = OperationJournal(ws)
    entry = journal.load(result.op_ids[0])
    notes_blob = " ".join(entry.notes)
    assert "render skipped at caller request (no_render=True)" in notes_blob


# ---------------------------------------------------------------------
# 8. CLI JSON shape and exit codes
# ---------------------------------------------------------------------


def test_cli_seed_apply_emits_structured_json_and_invalid_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload()
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    rc = cli_main(
        [
            "--root",
            str(tmp_path),
            "seed",
            "apply",
            str(manifest_path),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    payload_out = json.loads(captured.out)
    assert payload_out["dry_run"] is True
    assert payload_out["refusal_reason"] is None
    assert payload_out["claims_created"] == []
    assert len(payload_out["claims_planned"]) == 1
    planned = payload_out["claims_planned"][0]
    assert planned["claim_id"] == "c.alpha.1"
    assert planned["entity_id"] == "concept.alpha"
    assert planned["status"] == "draft"
    assert planned["source_id"] == "src.alpha"

    # Invalid manifest exits 1 with the structured refusal.
    bad_payload = _good_manifest_payload()
    bad_payload["version"] = "wrong_version"
    bad_manifest = _write_manifest(tmp_path / "bad.yaml", bad_payload)
    rc_bad = cli_main(
        ["--root", str(tmp_path), "seed", "apply", str(bad_manifest)]
    )
    captured_bad = capsys.readouterr()
    assert rc_bad == 1
    bad_out = json.loads(captured_bad.out)
    assert bad_out["refusal_reason"]
    assert "wrong_version" in bad_out["refusal_reason"]


def test_invalid_cli_status_refuses_before_persistence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload()
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    rc = cli_main(
        [
            "--root",
            str(tmp_path),
            "seed",
            "apply",
            str(manifest_path),
            "--status",
            "bogus_state",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    out = json.loads(captured.out)
    assert out["refusal_reason"]
    assert "bogus_state" in out["refusal_reason"]

    # No source registered, no claim, no journal entry.
    assert SourceRegistry(ws).list_ids() == []
    assert not ClaimStore(ws).exists("concept.alpha")
    assert _journal_count(ws, op_kind="seed_apply") == 0


# ---------------------------------------------------------------------
# Slice 075a follow-up: source-registration preflight
# ---------------------------------------------------------------------


def _assert_zero_workspace_mutation(ws: Workspace) -> None:
    """Helper: every Slice 075a refusal must leave the workspace
    byte-clean of seed-apply effects.
    """
    assert SourceRegistry(ws).list_ids() == [], (
        f"source registry was mutated: {SourceRegistry(ws).list_ids()}"
    )
    assert not ClaimStore(ws).exists("concept.alpha"), (
        "claim YAML was written despite refusal"
    )
    assert _journal_count(ws, op_kind="seed_apply") == 0, (
        "seed_apply journal entry exists despite refusal"
    )
    assert not (ws.state_locks / "workspace.yaml").is_file(), (
        "workspace lock was acquired and never released"
    )


def test_invalid_source_id_refuses_before_operation(tmp_path: Path) -> None:
    """Slice 075a: a manifest whose merged ``source_id`` does not
    match the registry id rule must refuse with a structured
    ``refusal_reason`` and zero workspace mutation, **without**
    ``apply_seed_manifest`` raising.
    """
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload()
    payload["sources"][0]["source_id"] = "BAD ID"
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    # apply_seed_manifest must not raise — it must return a
    # structured refusal.
    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded is False
    assert result.refusal_reason
    lowered = result.refusal_reason.lower()
    assert "source_id" in lowered or "invalid source_id" in lowered
    assert "bad id" in lowered or "'bad id'" in lowered.replace('"', "'")

    _assert_zero_workspace_mutation(ws)


def test_unknown_source_class_refuses_before_operation(tmp_path: Path) -> None:
    """Slice 075a: a manifest whose merged ``source_class`` is not
    declared in ``schema/source_classes.yaml`` must refuse with a
    structured ``refusal_reason`` naming the unknown class.
    """
    ws = _seed_workspace(tmp_path)
    payload = _good_manifest_payload()
    payload["defaults"]["source_class"] = "does_not_exist"
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", payload)

    result = apply_seed_manifest(ws, manifest_path)
    assert result.succeeded is False
    assert result.refusal_reason
    assert "unknown source_class" in result.refusal_reason
    assert "does_not_exist" in result.refusal_reason

    _assert_zero_workspace_mutation(ws)


def test_existing_source_hash_conflict_refuses_before_operation(
    tmp_path: Path,
) -> None:
    """Slice 075a: a manifest reusing an existing ``source_id``
    whose on-disk file's hash no longer matches the registered
    content_hash must refuse with a structured ``refusal_reason``.
    """
    ws = _seed_workspace(tmp_path)
    # Register the source once via a successful seed apply.
    payload = _good_manifest_payload()
    first_manifest = _write_manifest(tmp_path / "manifest1.yaml", payload)
    first = apply_seed_manifest(ws, first_manifest)
    assert first.succeeded, first.refusal_reason
    journals_before = _journal_count(ws, op_kind="seed_apply")
    assert journals_before == 1

    # Mutate the on-disk source bytes (raw evidence). The registry's
    # immutability rule should refuse a re-register with the new
    # hash via the preflight, NOT via SourceRegistryError inside
    # operation(...).
    src = ws.raw_sources / "alpha.md"
    src.write_text(SOURCE_TEXT + "\nExtra trailing line introduced after first apply.\n", encoding="utf-8")

    # Apply a second manifest with the same id.
    second_payload = _good_manifest_payload()
    second_manifest = _write_manifest(tmp_path / "manifest2.yaml", second_payload)
    second = apply_seed_manifest(ws, second_manifest)

    assert second.succeeded is False
    assert second.refusal_reason
    lowered = second.refusal_reason.lower()
    assert "hash" in lowered
    assert "raw evidence" in lowered or "immutable" in lowered

    # Exactly one prior seed_apply journal entry remains; no new
    # entry was opened.
    assert _journal_count(ws, op_kind="seed_apply") == journals_before
    assert not (ws.state_locks / "workspace.yaml").is_file()


def test_existing_source_class_conflict_refuses_before_operation(
    tmp_path: Path,
) -> None:
    """Slice 075a: a manifest reusing an existing ``source_id`` with
    a different ``source_class`` must refuse with a structured
    ``refusal_reason`` (the registry's class-immutability rule
    fires during preflight, not inside ``operation(...)``).
    """
    ws = _seed_workspace(tmp_path)
    # Register the source once with source_class=markdown_prose.
    payload = _good_manifest_payload()
    first_manifest = _write_manifest(tmp_path / "manifest1.yaml", payload)
    first = apply_seed_manifest(ws, first_manifest)
    assert first.succeeded, first.refusal_reason
    journals_before = _journal_count(ws, op_kind="seed_apply")
    assert journals_before == 1

    # Apply a second manifest with the same id but legal_act class.
    second_payload = _good_manifest_payload()
    second_payload["defaults"]["source_class"] = "legal_act"
    # Match the legal_act locator so we exercise only the
    # source_class preflight check, not a downstream locator type
    # mismatch.
    second_payload["defaults"]["locator"]["locator_type"] = "legal_act_v1"
    second_manifest = _write_manifest(tmp_path / "manifest2.yaml", second_payload)
    second = apply_seed_manifest(ws, second_manifest)

    assert second.succeeded is False
    assert second.refusal_reason
    lowered = second.refusal_reason.lower()
    assert "class mismatch" in lowered
    assert "markdown_prose" in lowered
    assert "legal_act" in lowered

    # No new seed_apply journal entry; lock not acquired.
    assert _journal_count(ws, op_kind="seed_apply") == journals_before
    assert not (ws.state_locks / "workspace.yaml").is_file()
