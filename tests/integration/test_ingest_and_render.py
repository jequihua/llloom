"""Integration: ingest a synthetic scientific source, render a concept
page with variant-(B) semantics, and verify round-trip via `verify`."""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.verify import verify
from llloom.pages.regions import parse_page
from llloom.workspace.layout import Workspace


PAGE_TEMPLATE = """\
---
page_id: concept/complementarity
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.concept.complementarity -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.complementarity owner=human -->

Human commentary that must be preserved across rerender.

<!-- /llloom:commentary -->
"""

SOURCE_TEXT = """\
# Article

## Methods

Complementarity prioritizes sites that add features not already represented in the selected set. It is commonly used to identify gaps.

Second paragraph irrelevant to the claim.
"""


def _seed_pages_and_source(ws: Workspace) -> Path:
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    page_path = ws.pages / "concepts" / "complementarity.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    return src


def test_ingest_seeds_claim_and_renders_page(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    src = _seed_pages_and_source(ws)
    seed = SeedClaim(
        entity_id="concept.complementarity",
        entity_type="concept",
        display_name="Complementarity",
        claim_id="c_0001",
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
        render_target=("concept/complementarity", "claim_block.concept.complementarity"),
    )
    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[seed],
    )
    assert result.succeeded
    assert "c_0001" in {c.claim_id for c in result.claims_created}
    assert "concept.complementarity" in result.entities_touched
    rendered = [p for p in result.pages_rendered if p.endswith("complementarity.md")]
    assert rendered, f"expected concept page rendered, got {result.pages_rendered}"

    # Commentary region must be preserved byte-for-byte.
    page_text = (ws.pages / "concepts" / "complementarity.md").read_text(encoding="utf-8")
    parsed = parse_page(page_text)
    assert "Human commentary that must be preserved" in parsed.commentary_inner
    assert "claim:c_0001" in parsed.claim_block_inner
    assert "Complementarity" in parsed.claim_block_inner

    # verify should pass
    v = verify(ws)
    assert v.passed, v.notes


def test_ingest_refuses_empty_source(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "empty.md"
    src.write_text("   \n\n", encoding="utf-8")
    result = ingest(ws, src, source_id="src.empty", source_class="markdown_prose")
    assert not result.succeeded
    assert result.refusal_reason == "empty source"


def test_ingest_denied_class_refused(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    # Declare a denied source class via schema edit.
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  denied_fixture:\n"
        "    locator: markdown_prose_v1\n",
        encoding="utf-8",
    )
    (ws.schema / "ingest_policies.yaml").write_text(
        "policies:\n"
        "  markdown_prose: claim_extract_and_view_render\n"
        "  denied_fixture: deny\n"
        "defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )
    src = ws.raw_sources / "credentials.md"
    src.write_text("FAKE_API_KEY=sk-test-0000", encoding="utf-8")
    result = ingest(
        ws, src, source_id="src.credentials", source_class="denied_fixture"
    )
    assert not result.succeeded
    assert result.refusal_reason == "policy deny"


def test_ingest_index_only_registers_without_claims(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  sensitive:\n"
        "    locator: markdown_prose_v1\n",
        encoding="utf-8",
    )
    (ws.schema / "ingest_policies.yaml").write_text(
        "policies:\n"
        "  markdown_prose: claim_extract_and_view_render\n"
        "  sensitive: index_only\n"
        "defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )
    src = ws.raw_sources / "contract.md"
    src.write_text("# Vendor contract\n\nPayment terms: net-30, 2% discount.\n", encoding="utf-8")
    result = ingest(ws, src, source_id="src.contract", source_class="sensitive")
    assert result.succeeded
    assert result.registration_state == "new"
    assert result.policy == "index_only"
    assert result.claims_created == []
    assert result.pages_rendered == []

