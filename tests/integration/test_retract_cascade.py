"""Integration: retract a source and observe the cascade."""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.claims.store import ClaimStore
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.retract import retract
from llloom.workspace.layout import Workspace


PAGE = """\
---
page_id: concept/foo
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.concept.foo -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.foo owner=human -->

commentary

<!-- /llloom:commentary -->
"""

SOURCE = """\
# Doc

## Methods

A simple fact sentence about foo. Second sentence.
"""


def _ingest_claim(ws: Workspace, claim_status: str = "draft") -> None:
    src = ws.raw_sources / "doc.md"
    src.write_text(SOURCE, encoding="utf-8")
    page = ws.pages / "concepts" / "foo.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(PAGE, encoding="utf-8")
    seed = SeedClaim(
        entity_id="concept.foo",
        entity_type="concept",
        display_name="Foo",
        claim_id="c_foo",
        claim_kind="fact",
        claim_text="A simple fact sentence about foo.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("concept/foo", "claim_block.concept.foo"),
        status=claim_status,
    )
    ingest(
        ws,
        src,
        source_id="src.doc",
        source_class="markdown_prose",
        seed_claims=[seed],
    )


def test_retract_draft_claim_marks_retracted_by_source(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    _ingest_claim(ws, claim_status="draft")
    result = retract(ws, source_id="src.doc", reason="test retraction")
    assert "c_foo" in result.affected_claim_ids
    store = ClaimStore(ws)
    entity = store.load_entity("concept.foo")
    assertion = entity.find_assertion("c_foo")
    assert assertion is not None
    assert assertion.status == "retracted_by_source"
    # Tombstone exists and is under state/reports/health.
    tombstone = ws.root / result.tombstone_path
    assert tombstone.is_file()
    # Affected pages are reported and rerendered.
    assert any(p.endswith("foo.md") for p in result.affected_pages)
    assert any(p.endswith("foo.md") for p in result.rerendered_pages)

