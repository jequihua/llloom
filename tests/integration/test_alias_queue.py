"""Integration: write-as-new, queue-for-merge alias semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.ops.alias import (
    list_merge_proposals,
    merge_alias,
    reject_alias,
    review_alias,
)
from llloom.ops.ingest import SeedClaim, ingest
from llloom.workspace.layout import Workspace


SOURCE_A = """\
# Doc A

## Intro

Andreas Gohr is the maintainer of DokuWiki. He writes on wiki governance.
"""

SOURCE_B = """\
# Doc B

## Intro

A. Gohr, writing for the DokuWiki newsletter, argues for editorial ownership.
"""


def _ingest_alias_scenario(ws: Workspace) -> None:
    src_a = ws.raw_sources / "doc_a.md"
    src_b = ws.raw_sources / "doc_b.md"
    src_a.write_text(SOURCE_A, encoding="utf-8")
    src_b.write_text(SOURCE_B, encoding="utf-8")

    seed_a = SeedClaim(
        entity_id="person.andreas_gohr",
        entity_type="person",
        display_name="Andreas Gohr",
        claim_id="c.gohr.1",
        claim_kind="role",
        claim_text="Andreas Gohr is the maintainer of DokuWiki.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Intro"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
    )
    ingest(
        ws,
        src_a,
        source_id="src.doc_a",
        source_class="markdown_prose",
        seed_claims=[seed_a],
    )

    seed_b = SeedClaim(
        entity_id="person.a_gohr",  # different entity_id on purpose
        entity_type="person",
        display_name="A. Gohr",
        claim_id="c.gohr.2",
        claim_kind="role",
        claim_text="A. Gohr, writing for the DokuWiki newsletter, argues for editorial ownership.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Intro"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
    )
    ingest(
        ws,
        src_b,
        source_id="src.doc_b",
        source_class="markdown_prose",
        seed_claims=[seed_b],
    )


def test_variant_spelling_does_not_auto_merge_but_queues(tmp_path: Path) -> None:
    """By-design: variant spellings ('A. Gohr' vs 'Andreas Gohr') do NOT
    match the first-slice alias matcher, which is strict on normalized
    display name. No silent merge; and any future matching writes a
    proposal rather than merging inline. This test verifies the
    write-as-new contract: both entities exist after ingest."""
    ws = Workspace.init(tmp_path)
    _ingest_alias_scenario(ws)
    from llloom.claims.store import ClaimStore

    store = ClaimStore(ws)
    ids = set(store.list_entity_ids())
    assert "person.andreas_gohr" in ids
    assert "person.a_gohr" in ids


def test_identical_display_name_queues_merge_proposal(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "doc.md"
    src.write_text(SOURCE_A, encoding="utf-8")
    seed1 = SeedClaim(
        entity_id="person.andreas_gohr",
        entity_type="person",
        display_name="Andreas Gohr",
        claim_id="c1",
        claim_kind="role",
        claim_text="Andreas Gohr is the maintainer of DokuWiki.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Intro"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
    )
    ingest(ws, src, source_id="src.a", source_class="markdown_prose", seed_claims=[seed1])
    # Second ingest under a different entity_id but same display_name.
    src2 = ws.raw_sources / "doc2.md"
    src2.write_text(SOURCE_A, encoding="utf-8")  # reuse text; same span
    seed2 = SeedClaim(
        entity_id="person.gohr_v2",
        entity_type="person",
        display_name="Andreas Gohr",
        claim_id="c2",
        claim_kind="role",
        claim_text="Andreas Gohr is the maintainer of DokuWiki.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Intro"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
    )
    result = ingest(
        ws, src2, source_id="src.b", source_class="markdown_prose", seed_claims=[seed2]
    )
    assert result.merge_proposals_created, "expected a merge proposal"
    proposals = list_merge_proposals(ws)
    pid = proposals[0].proposal_id

    # Approve and apply the merge.
    summary = review_alias(ws, proposal_id=pid, decision="approve")
    assert summary.status == "approved"
    applied = merge_alias(ws, proposal_id=pid)
    assert applied.status == "applied"

    # Target now carries the alias; source is merged_into.
    from llloom.claims.store import ClaimStore

    store = ClaimStore(ws)
    target = store.load_entity("person.andreas_gohr")
    alias_texts = [a.alias_text for a in target.aliases]
    assert "Andreas Gohr" in alias_texts
    source = store.load_entity("person.gohr_v2")
    assert source.status == "merged_into"


def test_reject_alias_closes_proposal(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "doc.md"
    src.write_text(SOURCE_A, encoding="utf-8")
    # Create an initial entity + then a clashing second one via the same path.
    seed1 = SeedClaim(
        entity_id="person.a",
        entity_type="person",
        display_name="Andreas Gohr",
        claim_id="c1",
        claim_kind="role",
        claim_text="Andreas Gohr is the maintainer of DokuWiki.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Intro"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
    )
    ingest(ws, src, source_id="s.a", source_class="markdown_prose", seed_claims=[seed1])
    src2 = ws.raw_sources / "doc2.md"
    src2.write_text(SOURCE_A, encoding="utf-8")
    seed2 = SeedClaim(
        entity_id="person.b",
        entity_type="person",
        display_name="Andreas Gohr",
        claim_id="c2",
        claim_kind="role",
        claim_text="Andreas Gohr is the maintainer of DokuWiki.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Intro"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
    )
    r = ingest(ws, src2, source_id="s.b", source_class="markdown_prose", seed_claims=[seed2])
    pid = r.merge_proposals_created[0]
    s = reject_alias(ws, proposal_id=pid, notes="spurious")
    assert s.status == "rejected"

