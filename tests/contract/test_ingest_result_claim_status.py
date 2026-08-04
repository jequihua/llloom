"""Contract: ingest results carry structured `CreatedClaim` records.

Pins the Slice 070 contract from
``feedback/2026-05-22_llloom_development_roadmap_synthesis.md``:

- ``IngestResult.claims_created`` is ``list[CreatedClaim]`` carrying
  ``claim_id``, ``entity_id``, ``status``, and ``verification_status``;
- deterministic seed claims default to lifecycle ``"draft"`` unless
  the operation-level ``seed_claim_status`` kwarg names a different
  valid status;
- a per-candidate ``SeedClaim.status`` override beats
  ``seed_claim_status``;
- invalid ``seed_claim_status`` refuses the batch atomically before
  any claim, entity, page, fingerprint, or merge proposal is written;
- model-backed candidates carry the persisted status (always set by
  the parser; never silently rewritten by ``seed_claim_status``) and
  surface as ``CreatedClaim`` records with
  ``verification_status="verified"``;
- the CLI ``ingest`` JSON output renders ``claims_created`` as a list
  of objects with the structured fields, not a list of strings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.claims.store import ClaimStore
from llloom.cli import main as cli_main
from llloom.llm.harness import LLMInvoke
from llloom.ops import CreatedClaim
from llloom.ops.ingest import SeedClaim, ingest
from llloom.workspace.layout import Workspace


SOURCE_TEXT = """\
# Article

## Methods

Complementarity prioritizes sites that add features not already represented in the selected set.
"""


def _seed_workspace(tmp_path: Path) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    return ws, src


def _seed_claim(claim_id: str = "c.cmp.1", *, status: str | None = None) -> SeedClaim:
    return SeedClaim(
        entity_id="concept.complementarity",
        entity_type="concept",
        display_name="Complementarity",
        claim_id=claim_id,
        claim_kind="definition",
        claim_text=(
            "Complementarity prioritizes sites that add features not already "
            "represented in the selected set."
        ),
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        status=status,
    )


class _FakeModel:
    identifier = "fake-test-model/v0"

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, prompt: str) -> str:
        _ = prompt
        return self._output


_MODEL_OUTPUT = """\
claims:
  - entity_id: concept.complementarity
    entity_type: concept
    display_name: Complementarity
    claim_id: c.model.1
    claim_kind: definition
    claim_text: |-
      Complementarity prioritizes sites that add features not already
      represented in the selected set.
    locator:
      locator_type: markdown_prose_v1
      heading_path: ["Methods"]
      paragraph_index: 1
      sentence_start: 1
      sentence_end: 1
"""


def test_seed_claims_default_to_draft(tmp_path: Path) -> None:
    ws, src = _seed_workspace(tmp_path)
    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[_seed_claim()],
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert len(result.claims_created) == 1
    rec = result.claims_created[0]
    assert isinstance(rec, CreatedClaim)
    assert rec.claim_id == "c.cmp.1"
    assert rec.entity_id == "concept.complementarity"
    assert rec.status == "draft"
    assert rec.verification_status == "verified"

    persisted = (
        ClaimStore(ws)
        .load_entity("concept.complementarity")
        .find_assertion("c.cmp.1")
    )
    assert persisted is not None
    assert persisted.status == "draft"
    assert persisted.verification_status == "verified"


def test_seed_claim_status_reviewed_lands_at_reviewed(tmp_path: Path) -> None:
    ws, src = _seed_workspace(tmp_path)
    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[_seed_claim()],
        seed_claim_status="reviewed",
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    rec = result.claims_created[0]
    assert rec.status == "reviewed"
    assert rec.verification_status == "verified"

    persisted = (
        ClaimStore(ws)
        .load_entity("concept.complementarity")
        .find_assertion("c.cmp.1")
    )
    assert persisted is not None
    assert persisted.status == "reviewed"
    assert persisted.verification_status == "verified"


def test_seed_claim_explicit_status_still_wins(tmp_path: Path) -> None:
    ws, src = _seed_workspace(tmp_path)
    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[_seed_claim(status="validated")],
        seed_claim_status="reviewed",
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    rec = result.claims_created[0]
    assert rec.status == "validated", (
        "explicit per-candidate status must win over operation-level default"
    )

    persisted = (
        ClaimStore(ws)
        .load_entity("concept.complementarity")
        .find_assertion("c.cmp.1")
    )
    assert persisted is not None
    assert persisted.status == "validated"


def test_seed_claim_status_invalid_refuses_batch_atomically(tmp_path: Path) -> None:
    ws, src = _seed_workspace(tmp_path)
    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[_seed_claim()],
        seed_claim_status="bogus_state",
    )
    assert result.succeeded is False
    assert result.refusal_reason is not None
    assert "bogus_state" in result.refusal_reason
    assert any("bogus_state" in line for line in result.extraction_errors)

    # Batch-atomic refusal: no claim, entity, page, fingerprint, or
    # merge proposal landed on disk from this batch.
    assert result.claims_created == []
    assert result.entities_touched == []
    assert result.pages_rendered == []
    assert result.merge_proposals_created == []

    store = ClaimStore(ws)
    assert not store.exists("concept.complementarity")
    # No render fingerprint store written (the file is created lazily).
    assert not (ws.state / "fingerprints" / "render.json").is_file()


def test_model_backed_created_claims_report_status_and_verification(
    tmp_path: Path,
) -> None:
    ws, src = _seed_workspace(tmp_path)
    fake = _FakeModel(_MODEL_OUTPUT)
    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=fake),
        # Even with a non-default seed_claim_status the model candidate
        # must keep its parser-supplied status; the kwarg only applies
        # to deterministic seed claims that opt in.
        seed_claim_status="reviewed",
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert len(result.claims_created) == 1
    rec = result.claims_created[0]
    assert rec.claim_id == "c.model.1"
    assert rec.entity_id == "concept.complementarity"
    # Parser default for model output is "draft"; seed_claim_status="reviewed"
    # must NOT silently promote it.
    assert rec.status == "draft"
    assert rec.verification_status == "verified"


def test_cli_ingest_outputs_created_claim_objects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Use the model-backed CLI path so we don't need a SeedClaim CLI
    # surface (which is deferred to Slice 075). The CLI exercises the
    # same _dc(...) recursion used for every other dataclass result.
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")

    from unittest.mock import patch

    fake = _FakeModel(_MODEL_OUTPUT)

    # Bypass the CLI's provider-import gate by patching _build_harness
    # to return a pre-built LLMInvoke around our fake. This keeps the
    # offline test suite contract intact.
    with patch("llloom.cli._build_harness", return_value=LLMInvoke(model=fake)):
        rc = cli_main(
            [
                "--root",
                str(tmp_path),
                "ingest",
                str(src),
                "--source-id",
                "src.article",
                "--source-class",
                "markdown_prose",
            ]
        )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    created = payload["claims_created"]
    assert isinstance(created, list)
    assert len(created) == 1
    obj = created[0]
    assert isinstance(obj, dict), f"expected dict, got {type(obj).__name__}"
    assert obj == {
        "claim_id": "c.model.1",
        "entity_id": "concept.complementarity",
        "status": "draft",
        "verification_status": "verified",
    }
