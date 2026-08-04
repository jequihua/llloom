"""Integration tests: ``query`` with the search sidecar present.

The sidecar must (a) improve retrieval where the naive token scorer is
weak, (b) preserve the existing ``QueryResult`` shape, and (c) remain
optional — if the sidecar is absent, ``query`` falls back to the
deterministic canonical walk without any change in behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.query import query
from llloom.ops.rebuild import rebuild
from llloom.ops.results import QueryResult, VerbatimSpan
from llloom.state.search import sidecar_exists
from llloom.workspace.layout import Workspace


# A claim whose canonical text contains a diacritic. The naive scorer
# lowercases and substring-matches, so `"cafe"` is NOT in `"café"` as a
# substring (`é` and `e` are distinct code points). FTS5 with
# ``remove_diacritics=1`` folds both to the same token, so the sidecar
# can rescue this hit.
DIACRITIC_SOURCE = """\
# Article

## Methods

Café culture shapes morning meetings in Madrid and Buenos Aires.
"""

DIACRITIC_PAGE = """\
---
page_id: concept/cafe
page_class: concept
write_policy: mixed
status: rendered
---

<!-- llloom:claim-block id=claim_block.concept.cafe -->
Original block.
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.cafe owner=human -->
Commentary.
<!-- /llloom:commentary -->
"""


def _seed_diacritic_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(DIACRITIC_SOURCE, encoding="utf-8")
    page_path = ws.pages / "concepts" / "cafe.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(DIACRITIC_PAGE, encoding="utf-8")

    seed = SeedClaim(
        entity_id="concept.cafe",
        entity_type="concept",
        display_name="Cafe",
        claim_id="c.cafe.1",
        claim_kind="definition",
        claim_text="Café culture shapes morning meetings in Madrid and Buenos Aires.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("concept/cafe", "claim_block.concept.cafe"),
    )
    ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[seed],
    )
    return ws


def test_query_without_sidecar_misses_diacritic_match(tmp_path: Path) -> None:
    """Fallback path contract: with no sidecar, ``query`` behavior is
    unchanged — so the diacritic-laden claim is missed by the naive
    substring scorer when the query uses the ASCII form."""
    ws = _seed_diacritic_workspace(tmp_path)
    assert not sidecar_exists(ws)

    result = query(ws, question="cafe")
    assert isinstance(result, QueryResult)
    # Naive path does not fold diacritics; "cafe" is not a substring of
    # "café" and this claim should not surface.
    assert "c.cafe.1" not in result.used_claim_ids


def test_query_with_sidecar_rescues_diacritic_match(tmp_path: Path) -> None:
    """With the sidecar built, FTS5 ``remove_diacritics=1`` makes the
    claim reachable through the ASCII query form. The rehydrated
    citation still comes from the canonical claim YAML, so the
    emitted ``claim_text`` retains the diacritic."""
    ws = _seed_diacritic_workspace(tmp_path)
    rebuild(ws, target="search")
    assert sidecar_exists(ws)

    result = query(ws, question="cafe")
    assert "c.cafe.1" in result.used_claim_ids
    assert result.citations, "expected at least one citation"
    cite = result.citations[0]
    assert cite["claim_id"] == "c.cafe.1"
    # Critical: the rehydrated citation text comes from canonical YAML,
    # not the sidecar row. The diacritic is preserved.
    assert "Café" in cite["text"]


def test_query_result_shape_unchanged_with_sidecar(tmp_path: Path) -> None:
    """QueryResult surface is stable: same fields, same dataclass.
    Running through the sidecar path does not add or remove fields."""
    ws = _seed_diacritic_workspace(tmp_path)
    rebuild(ws, target="search")

    result = query(ws, question="cafe")
    # Fields promised by the public surface.
    assert hasattr(result, "question")
    assert hasattr(result, "answer")
    assert hasattr(result, "citations")
    assert hasattr(result, "used_claim_ids")
    assert hasattr(result, "used_verbatim_spans")
    assert hasattr(result, "used_structure_items")
    assert isinstance(result.used_verbatim_spans, list)
    assert isinstance(result.used_structure_items, list)
    for span in result.used_verbatim_spans:
        assert isinstance(span, VerbatimSpan)


def test_query_fallback_without_sidecar_still_returns_native_hits(
    tmp_path: Path,
) -> None:
    """Counterpoint to the sidecar-rescue test: an ASCII-only claim is
    still retrievable through the native path with no sidecar, so the
    fallback path is genuinely exercised, not silently dead."""
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "plain.md"
    src.write_text(
        "# Plain\n\n## Methods\n\nAlpha beta gamma delta epsilon zeta.\n",
        encoding="utf-8",
    )
    page_path = ws.pages / "concepts" / "alpha.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(
        "---\npage_id: concept/alpha\npage_class: concept\n"
        "write_policy: mixed\nstatus: rendered\n---\n\n"
        "<!-- llloom:claim-block id=claim_block.concept.alpha -->\n"
        "x\n"
        "<!-- /llloom:claim-block -->\n\n"
        "<!-- llloom:commentary id=commentary.concept.alpha owner=human -->\n"
        "y\n"
        "<!-- /llloom:commentary -->\n",
        encoding="utf-8",
    )
    ingest(
        ws,
        src,
        source_id="src.plain",
        source_class="markdown_prose",
        seed_claims=[
            SeedClaim(
                entity_id="concept.alpha",
                entity_type="concept",
                display_name="Alpha",
                claim_id="c.alpha.1",
                claim_kind="definition",
                claim_text="Alpha beta gamma delta epsilon zeta.",
                locator=Locator(
                    locator_type="markdown_prose_v1",
                    heading_path=["Methods"],
                    paragraph_index=1,
                    sentence_start=1,
                    sentence_end=1,
                ),
                render_target=("concept/alpha", "claim_block.concept.alpha"),
            )
        ],
    )
    assert not sidecar_exists(ws)

    result = query(ws, question="gamma delta")
    assert "c.alpha.1" in result.used_claim_ids


# --- structure-report search-sidecar integration tests ------------------


_POLICIES_YAML = (
    "policies:\n"
    "  markdown_prose: claim_extract_and_view_render\n"
    "  legal_act: claim_extract\n"
    "defaults:\n"
    "  unknown: deny\n"
)


def _seed_structured_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "policies.yaml"
    src.write_text(_POLICIES_YAML, encoding="utf-8")
    ingest(ws, src, source_id="src.policies", source_class="structured_yaml")
    return ws


def test_query_with_search_sidecar_surfaces_structure_report_symbols(
    tmp_path: Path,
) -> None:
    """End-to-end: ingest a structured YAML source, rebuild the search
    sidecar, query for a symbol token, and get a rehydrated
    ``StructureItemHit`` plus a stable answer string."""
    ws = _seed_structured_workspace(tmp_path)
    rebuild(ws, target="search")
    result = query(ws, question="markdown_prose")

    assert result.used_structure_items, (
        "expected at least one StructureItemHit for a symbol token present "
        "in the structured source"
    )
    symbols = {it.symbol_path for it in result.used_structure_items}
    assert "policies.markdown_prose" in symbols
    match = next(
        it for it in result.used_structure_items if it.symbol_path == "policies.markdown_prose"
    )
    assert match.source_id == "src.policies"
    assert match.language == "yaml"
    assert match.kind == "mapping_key"
    assert match.locator["locator_type"] == "code_v1"
    assert match.report_path == "state/structure/src.policies.yaml"
    # Answer line is deterministic and references the item.
    assert "structure item" in result.answer
    assert "policies.markdown_prose" in result.answer


def test_query_result_shape_additively_includes_structure_items(
    tmp_path: Path,
) -> None:
    """``QueryResult`` carries the new ``used_structure_items`` list
    additively: every prior field (``citations``, ``used_claim_ids``,
    ``used_verbatim_spans``) is still present with its prior shape."""
    ws = _seed_structured_workspace(tmp_path)
    rebuild(ws, target="search")
    result = query(ws, question="markdown_prose")

    assert hasattr(result, "citations")
    assert isinstance(result.citations, list)
    assert hasattr(result, "used_claim_ids")
    assert isinstance(result.used_claim_ids, list)
    assert hasattr(result, "used_verbatim_spans")
    assert isinstance(result.used_verbatim_spans, list)
    for span in result.used_verbatim_spans:
        assert isinstance(span, VerbatimSpan)
    assert hasattr(result, "used_structure_items")
    assert isinstance(result.used_structure_items, list)
    from llloom.ops.results import StructureItemHit

    for hit in result.used_structure_items:
        assert isinstance(hit, StructureItemHit)
        assert isinstance(hit.locator, dict)


def test_search_sidecar_still_rescues_claim_hits_with_structure_rows_present(
    tmp_path: Path,
) -> None:
    """The diacritic-rescue behavior must still hold when a structure
    report is also in the workspace. The claim citation is still
    rehydrated from canonical claim YAML (the diacritic survives),
    and the structure path does not displace it."""
    ws = _seed_diacritic_workspace(tmp_path)
    # Also plant a structured YAML source so rebuild indexes
    # structure rows alongside claim rows.
    src = ws.raw_sources / "policies.yaml"
    src.write_text(_POLICIES_YAML, encoding="utf-8")
    ingest(ws, src, source_id="src.policies", source_class="structured_yaml")
    rebuild(ws, target="search")

    result = query(ws, question="cafe")
    # Claim rescue still works: the diacritic-bearing canonical text
    # surfaces through the sidecar hint even though the naive token
    # scorer does not match `cafe` against `Café` directly.
    assert result.used_claim_ids, "expected sidecar rescue of the Café claim"
    citation_texts = [c["text"] for c in result.citations]
    assert any("Café" in text for text in citation_texts)
