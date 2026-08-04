"""Contract test for the canary exclusion enforcement.

The fixed fixture canary token ``LLLOOM_CANARY_FIXED_Z9F3`` lives in a
page's commentary region and in the editorial-spine overview. A full
ingest + render cycle must produce zero occurrences of the token in
claim records, rendered claim-block regions, query answers, or
invocation-log output hashes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from llloom.claims.models import Locator
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.lint import FIXED_CANARY_TOKEN, lint
from llloom.ops.query import query
from llloom.workspace.layout import Workspace


TESTS_DIR = Path(__file__).resolve().parents[1]
CANARY_DIR = TESTS_DIR / "fixtures" / "synthetic" / "canary"


SOURCE_TEXT = """\
# Source

## Section

A plain fact sentence. A second sentence.
"""


def _seed(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    # Drop the canary page and spine overview into the workspace.
    page = ws.pages / "canary" / "leak_test.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CANARY_DIR / "canary_page.md", page)
    shutil.copyfile(
        CANARY_DIR / "spine_overview.md", ws.pages / "overview.md"
    )
    # Add a source.
    src = ws.raw_sources / "source.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    return ws


def test_canary_never_leaks_into_claim_records_or_rendered_blocks(tmp_path: Path) -> None:
    ws = _seed(tmp_path)
    seed = SeedClaim(
        entity_id="concept.leak",
        entity_type="concept",
        display_name="Leak Test",
        claim_id="c.leak.1",
        claim_kind="fact",
        claim_text="A plain fact sentence.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Section"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("canary/leak_test", "claim_block.canary.leak_test"),
    )
    src = ws.raw_sources / "source.md"
    ingest(ws, src, source_id="src.leak", source_class="markdown_prose", seed_claims=[seed])

    # Assert the token appears in commentary + spine, but nowhere else.
    page_text = (ws.pages / "canary" / "leak_test.md").read_text(encoding="utf-8")
    overview_text = (ws.pages / "overview.md").read_text(encoding="utf-8")
    assert FIXED_CANARY_TOKEN in page_text
    assert FIXED_CANARY_TOKEN in overview_text

    # Claim YAML must not contain the token.
    entity_path = ws.claims_entities / "concept.leak.yaml"
    assert FIXED_CANARY_TOKEN not in entity_path.read_text(encoding="utf-8")

    # Rendered claim-block must not contain the token.
    from llloom.pages.regions import parse_page

    parsed = parse_page(page_text)
    assert FIXED_CANARY_TOKEN not in parsed.claim_block_inner
    # The token lives only in the commentary region, where it is
    # non-authoritative and excluded from LLM context.
    assert FIXED_CANARY_TOKEN in parsed.commentary_inner

    # Query answer must not contain the token.
    q = query(ws, question="plain fact")
    assert FIXED_CANARY_TOKEN not in q.answer

    # Lint must pass: token appears only in commentary + spine, neither
    # is a forbidden observation point.
    result = lint(ws)
    assert not result.canary_hits


def test_lint_flags_canary_leak_into_claim_block(tmp_path: Path) -> None:
    """If a renderer (or a manual mutation) leaks the canary into a
    rendered claim-block, lint MUST flag it."""
    ws = _seed(tmp_path)
    # Place the fixed canary directly into a claim-block region by hand.
    page = ws.pages / "canary" / "leak_test.md"
    text = page.read_text(encoding="utf-8")
    poisoned = text.replace(
        "This claim block is populated",
        f"{FIXED_CANARY_TOKEN}\nThis claim block is populated",
    )
    page.write_text(poisoned, encoding="utf-8")
    result = lint(ws)
    assert result.canary_hits, "lint must flag canary leak in claim-block"
    assert any("claim-block" in h for h in result.canary_hits)

