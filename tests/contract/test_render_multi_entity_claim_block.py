"""Contract: multi-entity claim-block rendering (Slice 071).

Pins the structural fix to the WME Audio field-feedback bug. Before
Slice 071 the renderer planned ``(page_id, entity)`` items, so two
entities targeting the same page rendered twice and the later writer
overwrote the earlier claim block. After Slice 071 the renderer is
page/block-centric: every page renders at most once per operation
and the rendered claim block is the union of every render-visible
assertion targeting that page's ``(page_id, block_id)`` — ordered
by ``(entity_id, claim_id)`` for byte-stable output.

Covered:

1. Two entities → both contribute to the same rendered block.
2. Output is deterministic under different ingest orderings.
3. Each page id appears at most once in ``rendered_pages`` /
   ``unchanged_pages``.
4. Single-entity rendering is byte-compatible with the pre-Slice-071
   output shape.
5. Retracted / retracted_by_source / archived assertions are hidden;
   draft / reviewed / validated / stale / superseded markers remain.
6. Ingest-triggered rendering of one touched entity still renders
   the union for the affected page.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.claims.models import (
    Assertion,
    EntityContainer,
    Evidence,
    Locator,
    RenderTarget,
)
from llloom.claims.store import ClaimStore
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.render import render
from llloom.pages.regions import parse_page
from llloom.pages.render import (
    render_claim_block,
    render_claim_block_from_contributors,
    render_page_file,
)
from llloom.workspace.layout import Workspace


PAGE_TEMPLATE = """\
---
page_id: concept/shared
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.concept.shared -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.shared owner=human -->

Commentary that must survive byte-for-byte across rerender.

<!-- /llloom:commentary -->
"""


SOURCE_TEXT_ALPHA = """\
# Article

## Methods

Alpha entity asserts the first claim about shared architecture.

Second paragraph irrelevant.
"""

SOURCE_TEXT_BETA = """\
# Article

## Methods

Beta entity asserts the second claim about shared architecture.

Second paragraph irrelevant.
"""


def _seed_two_sources(ws: Workspace) -> tuple[Path, Path]:
    src_a = ws.raw_sources / "alpha.md"
    src_a.write_text(SOURCE_TEXT_ALPHA, encoding="utf-8")
    src_b = ws.raw_sources / "beta.md"
    src_b.write_text(SOURCE_TEXT_BETA, encoding="utf-8")
    page_path = ws.pages / "concepts" / "shared.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    return src_a, src_b


def _seed_claim_alpha() -> SeedClaim:
    return SeedClaim(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        claim_id="c.alpha.1",
        claim_kind="definition",
        claim_text=(
            "Alpha entity asserts the first claim about shared architecture."
        ),
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("concept/shared", "claim_block.concept.shared"),
    )


def _seed_claim_beta() -> SeedClaim:
    return SeedClaim(
        entity_id="concept.beta",
        entity_type="concept",
        display_name="Beta",
        claim_id="c.beta.1",
        claim_kind="definition",
        claim_text=(
            "Beta entity asserts the second claim about shared architecture."
        ),
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("concept/shared", "claim_block.concept.shared"),
    )


def _ingest_alpha_then_beta(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    src_a, src_b = _seed_two_sources(ws)
    ingest(
        ws,
        src_a,
        source_id="src.alpha",
        source_class="markdown_prose",
        seed_claims=[_seed_claim_alpha()],
    )
    ingest(
        ws,
        src_b,
        source_id="src.beta",
        source_class="markdown_prose",
        seed_claims=[_seed_claim_beta()],
    )
    return ws


def _ingest_beta_then_alpha(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    src_a, src_b = _seed_two_sources(ws)
    ingest(
        ws,
        src_b,
        source_id="src.beta",
        source_class="markdown_prose",
        seed_claims=[_seed_claim_beta()],
    )
    ingest(
        ws,
        src_a,
        source_id="src.alpha",
        source_class="markdown_prose",
        seed_claims=[_seed_claim_alpha()],
    )
    return ws


def _read_claim_block(ws: Workspace) -> str:
    text = (ws.pages / "concepts" / "shared.md").read_text(encoding="utf-8")
    parsed = parse_page(text)
    return parsed.claim_block_inner


def test_two_entities_one_block_renders_union(tmp_path: Path) -> None:
    ws = _ingest_alpha_then_beta(tmp_path)
    block_inner = _read_claim_block(ws)
    assert "claim:c.alpha.1" in block_inner, (
        "alpha entity's claim missing from union; multi-entity render bug"
    )
    assert "claim:c.beta.1" in block_inner, (
        "beta entity's claim missing from union; later ingest overwrote earlier"
    )
    # Both display names should appear as section headings (multi-entity).
    assert "## Alpha" in block_inner
    assert "## Beta" in block_inner
    # Commentary still preserved byte-for-byte.
    page_text = (ws.pages / "concepts" / "shared.md").read_text(encoding="utf-8")
    parsed = parse_page(page_text)
    assert "Commentary that must survive byte-for-byte" in parsed.commentary_inner


def test_render_order_is_deterministic_across_ingest_orderings(
    tmp_path: Path,
) -> None:
    ws_ab = _ingest_alpha_then_beta(tmp_path / "ab")
    ws_ba = _ingest_beta_then_alpha(tmp_path / "ba")
    block_ab = _read_claim_block(ws_ab)
    block_ba = _read_claim_block(ws_ba)
    assert block_ab == block_ba, (
        "rendered claim block is not byte-stable across ingest orderings"
    )
    # Entities sort by entity_id; alpha < beta lexicographically, so the
    # Alpha section must precede the Beta section regardless of ingest
    # ordering.
    alpha_idx = block_ab.index("## Alpha")
    beta_idx = block_ab.index("## Beta")
    assert alpha_idx < beta_idx
    # Re-rendering the same workspace should produce byte-identical
    # output (idempotence).
    result_again = render(ws_ab, target="page:concept/shared")
    # Page already up-to-date after ingest's render step → unchanged.
    assert any(p.endswith("shared.md") for p in result_again.unchanged_pages), (
        f"valid render should land in unchanged_pages on byte-stable "
        f"re-render; got rendered={result_again.rendered_pages} "
        f"unchanged={result_again.unchanged_pages}"
    )
    assert _read_claim_block(ws_ab) == block_ab


def test_result_lists_do_not_contain_duplicate_page_ids(tmp_path: Path) -> None:
    ws = _ingest_alpha_then_beta(tmp_path)
    # Render the whole workspace; the shared page is targeted by two
    # entities but must appear at most once across rendered_pages /
    # unchanged_pages.
    result = render(ws)
    counts = {
        path: result.rendered_pages.count(path)
        + result.unchanged_pages.count(path)
        for path in set(result.rendered_pages + result.unchanged_pages)
    }
    assert all(c == 1 for c in counts.values()), (
        f"page id appeared more than once across result lists: {counts}"
    )
    # Specifically the shared page is exactly once.
    shared_hits = [
        p for p in (result.rendered_pages + result.unchanged_pages)
        if p.endswith("shared.md")
    ]
    assert len(shared_hits) == 1


def test_single_entity_render_is_byte_compatible(tmp_path: Path) -> None:
    """A single-entity page produces the legacy heading + bullets shape.

    Renders the exact text Slice 068's preflight test exercised so the
    single-entity case stays byte-stable after Slice 071.
    """
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "alpha.md"
    src.write_text(SOURCE_TEXT_ALPHA, encoding="utf-8")
    page_path = ws.pages / "concepts" / "shared.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    ingest(
        ws,
        src,
        source_id="src.alpha",
        source_class="markdown_prose",
        seed_claims=[_seed_claim_alpha()],
    )
    block_inner = _read_claim_block(ws)
    # Legacy single-entity shape: one '## DisplayName' heading then the
    # bullet list. The framing newlines around the inner region come
    # from ``replace_claim_block`` and are unchanged by Slice 071.
    assert block_inner.strip().startswith("## Alpha")
    assert "## Beta" not in block_inner
    assert "claim:c.alpha.1" in block_inner
    assert "[DRAFT] Alpha entity asserts" in block_inner
    # No "no rendered assertions" sentinel — there is a claim.
    assert "_No rendered assertions._" not in block_inner


def test_render_visible_status_filter_hides_retracted_and_archived(
    tmp_path: Path,
) -> None:
    """Pin the renderer-only status filter.

    Constructs an entity carrying assertions across every lifecycle
    status by writing the entity YAML directly through ``ClaimStore``,
    then renders the page. Excluded statuses
    (``retracted``, ``retracted_by_source``, ``archived``) must not
    appear in the block; the rest must keep their legacy markers.
    """
    ws = Workspace.init(tmp_path)
    page_path = ws.pages / "concepts" / "shared.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")

    store = ClaimStore(ws)
    locator = Locator(
        locator_type="markdown_prose_v1",
        heading_path=["Methods"],
        paragraph_index=1,
        sentence_start=1,
        sentence_end=1,
    )

    def _assert(claim_id: str, status: str, text: str) -> Assertion:
        return Assertion(
            claim_id=claim_id,
            subject_id="concept.alpha",
            claim_kind="definition",
            claim_text=text,
            evidence=[
                Evidence(
                    source_id="src.fixture",
                    locator=locator,
                    excerpt_hash="sha256:placeholder",
                    excerpt=text,
                )
            ],
            render_targets=[
                RenderTarget(
                    page_id="concept/shared",
                    block_id="claim_block.concept.shared",
                )
            ],
            status=status,
            verification_status="verified",
            created_at="2026-05-22T00:00:00Z",
            updated_at="2026-05-22T00:00:00Z",
        )

    entity = EntityContainer(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        aliases=[],
        assertions=[
            _assert("c.draft", "draft", "Draft assertion text."),
            _assert("c.reviewed", "reviewed", "Reviewed assertion text."),
            _assert("c.validated", "validated", "Validated assertion text."),
            _assert("c.stale", "stale", "Stale assertion text."),
            _assert("c.superseded", "superseded", "Superseded assertion text."),
            _assert("c.retracted", "retracted", "Retracted should be hidden."),
            _assert(
                "c.retracted_by_source",
                "retracted_by_source",
                "Source-retracted should be hidden.",
            ),
            _assert("c.archived", "archived", "Archived should be hidden."),
        ],
        relations=[],
    )
    store.save_entity(entity)

    render(ws)
    block_inner = _read_claim_block(ws)

    # Visible statuses present with their legacy markers.
    assert "[DRAFT] Draft assertion" in block_inner
    assert "[REVIEWED] Reviewed assertion" in block_inner
    assert "[VALIDATED] Validated assertion" in block_inner
    assert "[STALE] Stale assertion" in block_inner
    assert "[SUPERSEDED] Superseded assertion" in block_inner

    # Hidden statuses fully absent — neither the marker nor the text.
    assert "[RETRACTED]" not in block_inner
    assert "[ARCHIVED]" not in block_inner
    assert "Retracted should be hidden" not in block_inner
    assert "Source-retracted should be hidden" not in block_inner
    assert "Archived should be hidden" not in block_inner


def _make_empty_alpha_entity() -> EntityContainer:
    """Single entity whose only block-targeting assertion is retracted.

    Used by the Slice 071a follow-up tests to exercise the
    empty-render-visible code path without invoking ingest.
    """
    locator = Locator(
        locator_type="markdown_prose_v1",
        heading_path=["Methods"],
        paragraph_index=1,
        sentence_start=1,
        sentence_end=1,
    )
    return EntityContainer(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        aliases=[],
        assertions=[
            Assertion(
                claim_id="c.alpha.retracted",
                subject_id="concept.alpha",
                claim_kind="definition",
                claim_text="Retracted assertion text — must be hidden by the filter.",
                evidence=[
                    Evidence(
                        source_id="src.fixture",
                        locator=locator,
                        excerpt_hash="sha256:placeholder",
                        excerpt="Retracted body.",
                    )
                ],
                render_targets=[
                    RenderTarget(
                        page_id="concept/shared",
                        block_id="claim_block.concept.shared",
                    )
                ],
                status="retracted",
                verification_status="verified",
                created_at="2026-05-22T00:00:00Z",
                updated_at="2026-05-22T00:00:00Z",
            ),
        ],
        relations=[],
    )


def test_legacy_single_entity_empty_block_preserves_heading_and_sentinel(
    tmp_path: Path,
) -> None:
    """Slice 071a follow-up: the legacy single-entity wrappers
    (``render_claim_block`` and ``render_page_file``) preserve the
    pre-Slice-071 empty shape
    ``## <display_name>\\n\\n_No rendered assertions._\\n``.

    The wrapper has unambiguous single-entity context — exactly one
    entity supplied — so it can name the heading even when zero
    render-visible assertions remain. The union helper (covered by
    the next test) has no such context and emits the bare sentinel
    on the empty path.
    """
    entity = _make_empty_alpha_entity()

    # Direct claim-block helper.
    block_text = render_claim_block(entity, "claim_block.concept.shared")
    assert block_text == "## Alpha\n\n_No rendered assertions._\n"

    # Page-file helper: the rendered claim-block region of the page
    # must contain both the entity heading and the sentinel line.
    ws = Workspace.init(tmp_path)
    page_path = ws.pages / "concepts" / "shared.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    result = render_page_file(ws, page_path, entity)
    assert result.wrote, "wrapper must rewrite the placeholder block"
    parsed = parse_page(page_path.read_text(encoding="utf-8"))
    assert "## Alpha" in parsed.claim_block_inner, (
        f"expected legacy heading in empty single-entity render; got "
        f"{parsed.claim_block_inner!r}"
    )
    assert "_No rendered assertions._" in parsed.claim_block_inner
    # Render-visible filter still applied — the retracted assertion's
    # body text must not leak into the rendered block.
    assert "Retracted assertion text" not in parsed.claim_block_inner
    # Commentary preserved byte-for-byte.
    assert "Commentary that must survive byte-for-byte" in parsed.commentary_inner
    # No claim ids reported because every targeting assertion is hidden.
    assert result.rendered_claim_ids == []


def test_union_helper_empty_block_returns_bare_sentinel_without_heading() -> None:
    """The union helper deliberately diverges from the wrapper on the
    empty path: zero render-visible contributors → bare sentinel,
    no heading. Pinning the divergence so it cannot drift back to
    a fake-heading shape (which would name an arbitrary contributor
    in the multi-entity empty case).
    """
    entity = _make_empty_alpha_entity()
    text = render_claim_block_from_contributors(
        [entity], "claim_block.concept.shared"
    )
    assert text == "_No rendered assertions._\n"
    assert "## Alpha" not in text


def test_ingest_one_entity_renders_union_for_shared_page(tmp_path: Path) -> None:
    """Ingest-triggered render of ONE entity must still include the
    OTHER entity's claims on the same page/block.

    This is the field-feedback core bug: WME Audio's second ingest
    overwrote the first entity's block because ingest re-rendered only
    the touched entity's contribution.
    """
    ws = Workspace.init(tmp_path)
    src_a, src_b = _seed_two_sources(ws)
    # First ingest seeds alpha and renders the page with alpha's claim.
    r1 = ingest(
        ws,
        src_a,
        source_id="src.alpha",
        source_class="markdown_prose",
        seed_claims=[_seed_claim_alpha()],
    )
    assert r1.succeeded
    # Second ingest touches only the beta entity. The ingest-triggered
    # render must produce a page that contains BOTH alpha and beta.
    r2 = ingest(
        ws,
        src_b,
        source_id="src.beta",
        source_class="markdown_prose",
        seed_claims=[_seed_claim_beta()],
    )
    assert r2.succeeded
    # The single touched entity is beta, but the rendered page must be
    # the union: alpha + beta.
    assert r2.entities_touched == ["concept.beta"]
    assert any(
        p.endswith("shared.md") for p in r2.pages_rendered
    ), f"ingest-triggered render should include the shared page: {r2.pages_rendered}"
    # And rendered exactly once per ingest op.
    shared_hits = [p for p in r2.pages_rendered if p.endswith("shared.md")]
    assert len(shared_hits) == 1

    block_inner = _read_claim_block(ws)
    assert "claim:c.alpha.1" in block_inner
    assert "claim:c.beta.1" in block_inner
    assert "## Alpha" in block_inner
    assert "## Beta" in block_inner
