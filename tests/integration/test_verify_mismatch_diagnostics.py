"""Integration tests: ``verify(...)`` and the model-extraction path
both surface structured ``VerifierMismatch`` diagnostics.

The hash contract is unchanged. Mismatches still fail. These tests
prove the diagnostics are present, structured, and bounded so an
agent or human can investigate without trawling raw source bytes
out of the workspace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.claims.verifier import VerifierMismatch
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.verify import verify
from llloom.workspace.layout import Workspace


SOURCE = """\
# Article

## Methods

A canonical sentence in paragraph one. Another sentence here.
"""


def _seed_workspace(tmp_path: Path) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE, encoding="utf-8")
    return ws, src


def _good_seed() -> SeedClaim:
    return SeedClaim(
        entity_id="concept.x",
        entity_type="concept",
        display_name="X",
        claim_id="c.x.1",
        claim_kind="definition",
        claim_text="A canonical sentence in paragraph one.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
    )


def test_verify_surfaces_structured_mismatch_after_source_drift(
    tmp_path: Path,
) -> None:
    """Persist a verified claim, mutate the source so the hash no
    longer matches, and confirm ``verify(...)`` returns a structured
    diagnostic on its public result."""
    ws, src = _seed_workspace(tmp_path)
    ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[_good_seed()],
    )

    # Mutate the raw source: change the paragraph so the cited span
    # resolves to different text and the hash check fails.
    mutated = SOURCE.replace(
        "A canonical sentence in paragraph one.",
        "A different sentence has now replaced the original.",
    )
    src.write_text(mutated, encoding="utf-8")

    result = verify(ws)
    assert not result.passed
    assert "c.x.1" in result.failed
    # Structured diagnostic on the public surface.
    assert result.mismatches, "expected at least one structured mismatch"
    m = result.mismatches[0]
    assert isinstance(m, VerifierMismatch)
    assert m.claim_id == "c.x.1"
    assert m.source_id == "src.article"
    assert m.locator_type == "markdown_prose_v1"
    assert m.stored_hash != m.computed_hash
    # Preview reflects the CURRENT (mutated) span text and is bounded.
    assert "different sentence" in m.current_preview.lower()
    assert len(m.current_preview) <= 120
    # Stored preview survives from the persisted evidence excerpt.
    assert m.stored_preview is not None
    assert "canonical sentence" in m.stored_preview


def test_extraction_error_includes_structured_diagnostic_text(
    tmp_path: Path,
) -> None:
    """A model candidate (or seed) with an explicitly wrong
    ``excerpt_hash`` refuses the whole batch, and the
    ``IngestResult.extraction_errors`` line carries the source id,
    locator type, both hashes, and a bounded preview."""
    ws, src = _seed_workspace(tmp_path)
    bad_seed = SeedClaim(
        entity_id="concept.x",
        entity_type="concept",
        display_name="X",
        claim_id="c.x.bad",
        claim_kind="definition",
        claim_text="A canonical sentence in paragraph one.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        excerpt_hash="sha256:" + "f" * 64,
    )
    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[bad_seed],
    )
    assert not result.succeeded
    assert "extraction failed" in result.refusal_reason
    assert result.extraction_errors, "expected per-candidate diagnostic text"
    note = result.extraction_errors[0]
    # The structured fields appear in the textual diagnostic.
    assert "c.x.bad" in note
    assert "source_id=src.article" in note
    assert "locator_type=markdown_prose_v1" in note
    assert "sha256:" + "f" * 64 in note  # stored
    assert "current_preview=" in note
    # Atomicity guard: nothing persisted.
    from llloom.claims.store import ClaimStore

    assert ClaimStore(ws).list_entity_ids() == []


def test_verify_clean_workspace_has_no_mismatches(tmp_path: Path) -> None:
    """Regression: a freshly-ingested, unmodified workspace verifies
    cleanly with an empty ``mismatches`` list."""
    ws, src = _seed_workspace(tmp_path)
    ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[_good_seed()],
    )
    result = verify(ws)
    assert result.passed
    assert result.failed == []
    assert result.mismatches == []
    assert result.notes == []
