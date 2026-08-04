"""Integration tests: model-backed `claim_extract` ingestion.

A deterministic fake model backend returns the strict YAML output
contract from ``llloom.llm.output``. Ingest parses the output, runs
each candidate through the verifier, and persists only verified
claims.

Covers:

- happy path: model output -> verified assertion under
  `claim_extract`
- `claim_extract_and_view_render`: target page is rendered and
  commentary survives byte-for-byte
- batch atomicity: a mixed-validity batch persists nothing
- `NullModel` (default) still produces zero claims without error
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.claims.store import ClaimStore
from llloom.llm.harness import LLMInvoke
from llloom.ops.ingest import ingest
from llloom.ops.verify import verify
from llloom.pages.regions import parse_page
from llloom.workspace.layout import Workspace


SOURCE = """\
# Article

## Methods

Complementarity prioritizes sites that add features not already represented in the selected set. It is widely used.

A second paragraph that mentions diversity but not the central concept.
"""


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

Human commentary that must survive rerender.

<!-- /llloom:commentary -->
"""


def _seed_workspace(tmp_path: Path, *, with_page: bool) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE, encoding="utf-8")
    if with_page:
        page_path = ws.pages / "concepts" / "complementarity.md"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    return ws, src


class _FakeModel:
    """Deterministic ModelBackend that returns a fixed string."""

    identifier = "fake-test-model/v0"

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, prompt: str) -> str:
        _ = prompt
        return self._output


_GOOD_OUTPUT = """\
claims:
  - entity_id: concept.complementarity
    entity_type: concept
    display_name: Complementarity
    claim_id: c.cmp.1
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


_GOOD_OUTPUT_WITH_RENDER_TARGET = """\
claims:
  - entity_id: concept.complementarity
    entity_type: concept
    display_name: Complementarity
    claim_id: c.cmp.1
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
    render_target: ["concept/complementarity", "claim_block.concept.complementarity"]
"""


def test_model_output_creates_verified_claim(tmp_path: Path) -> None:
    ws, src = _seed_workspace(tmp_path, with_page=False)
    fake = _FakeModel(_GOOD_OUTPUT)

    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=fake),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.cmp.1"]
    assert result.entities_touched == ["concept.complementarity"]

    store = ClaimStore(ws)
    entity = store.load_entity("concept.complementarity")
    assertion = entity.find_assertion("c.cmp.1")
    assert assertion is not None
    assert assertion.verification_status == "verified"
    assert assertion.evidence and assertion.evidence[0].source_id == "src.article"
    assert assertion.evidence[0].excerpt_hash.startswith("sha256:")
    assert assertion.evidence[0].excerpt is not None
    assert "Complementarity" in assertion.evidence[0].excerpt

    v = verify(ws)
    assert v.passed, v.notes


def test_claim_extract_and_view_render_renders_pages_and_preserves_commentary(
    tmp_path: Path,
) -> None:
    ws, src = _seed_workspace(tmp_path, with_page=True)
    fake = _FakeModel(_GOOD_OUTPUT_WITH_RENDER_TARGET)

    # Default starter schema maps `markdown_prose` -> claim_extract_and_view_render.
    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=fake),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.cmp.1"]
    rendered = [p for p in result.pages_rendered if p.endswith("complementarity.md")]
    assert rendered, f"expected page rendered, got {result.pages_rendered}"

    page_text = (ws.pages / "concepts" / "complementarity.md").read_text(encoding="utf-8")
    parsed = parse_page(page_text)
    # Commentary preserved byte-for-byte.
    assert "Human commentary that must survive rerender" in parsed.commentary_inner
    # Claim block contains the rendered claim with citation.
    assert "claim:c.cmp.1" in parsed.claim_block_inner
    assert "Complementarity" in parsed.claim_block_inner


def test_null_model_default_creates_zero_claims(tmp_path: Path) -> None:
    """The default NullModel returns an empty string. No claims, no error."""
    ws, src = _seed_workspace(tmp_path, with_page=False)
    result = ingest(
        ws, src, source_id="src.article", source_class="markdown_prose"
    )
    assert result.succeeded
    assert result.claims_created == []
    assert result.entities_touched == []
    assert result.extraction_errors == []
    assert ClaimStore(ws).list_entity_ids() == []


_MIXED_VALIDITY_BATCH = """\
claims:
  - entity_id: concept.complementarity
    entity_type: concept
    display_name: Complementarity
    claim_id: c.cmp.good
    claim_kind: definition
    claim_text: Complementarity is a method.
    locator:
      locator_type: markdown_prose_v1
      heading_path: ["Methods"]
      paragraph_index: 1
      sentence_start: 1
      sentence_end: 1
  - entity_id: concept.complementarity
    entity_type: concept
    display_name: Complementarity
    claim_id: c.cmp.bad
    claim_kind: definition
    claim_text: A bogus claim.
    locator:
      locator_type: markdown_prose_v1
      heading_path: ["Nonexistent Heading"]
      paragraph_index: 99
      sentence_start: 1
      sentence_end: 1
"""


def test_batch_with_one_bad_candidate_persists_nothing(tmp_path: Path) -> None:
    """Batch atomicity: one unresolvable locator refuses the whole batch."""
    ws, src = _seed_workspace(tmp_path, with_page=False)
    fake = _FakeModel(_MIXED_VALIDITY_BATCH)

    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=fake),
    )
    assert not result.succeeded
    assert "extraction failed" in result.refusal_reason
    assert any("c.cmp.bad" in note for note in result.extraction_errors)
    # Critical: the GOOD candidate also did not persist.
    assert ClaimStore(ws).list_entity_ids() == [], (
        "batch atomicity violated: a candidate from a failed batch persisted"
    )


def test_seed_claims_still_supported_alongside_model_output(tmp_path: Path) -> None:
    """Model + explicit seed candidates run through the same verifier."""
    from llloom.claims.models import Locator
    from llloom.ops.ingest import SeedClaim

    ws, src = _seed_workspace(tmp_path, with_page=False)
    fake = _FakeModel(_GOOD_OUTPUT)

    seed = SeedClaim(
        entity_id="concept.diversity",
        entity_type="concept",
        display_name="Diversity",
        claim_id="c.div.1",
        claim_kind="mention",
        claim_text="A second paragraph that mentions diversity but not the central concept.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=2,
            sentence_start=1,
            sentence_end=1,
        ),
    )
    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=fake),
        seed_claims=[seed],
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert sorted(c.claim_id for c in result.claims_created) == ["c.cmp.1", "c.div.1"]
    assert sorted(result.entities_touched) == [
        "concept.complementarity",
        "concept.diversity",
    ]
