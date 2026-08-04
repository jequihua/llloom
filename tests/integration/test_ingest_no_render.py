"""Integration tests for the ``ingest --no-render`` flag.

Slice contract:

- ``no_render=True`` on a ``claim_extract_and_view_render`` ingest
  persists verified claims but skips ``_render_targets``; the page
  claim-block is left exactly as it was on disk.
- The result reports ``render_skipped == True``,
  ``pages_rendered == []``, and ``succeeded == True``.
- Default behavior (no flag) still renders, proving the flag does
  not change the policy semantics for callers that omit it.
- The CLI passes the flag through; ``llloom ingest <path>
  --no-render`` produces JSON whose ``render_skipped`` field is
  ``true``.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.claims.store import ClaimStore
from llloom.cli import main as cli_main
from llloom.ops.ingest import SeedClaim, ingest
from llloom.pages.regions import parse_page
from llloom.workspace.layout import Workspace


SOURCE = """\
# Article

## Methods

Complementarity prioritizes sites that add features not already represented in the selected set.
"""


PAGE_TEMPLATE = """\
---
page_id: concept/complementarity
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.concept.complementarity -->
ORIGINAL_CLAIM_BLOCK_PLACEHOLDER
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.complementarity owner=human -->

Human commentary that must survive.

<!-- /llloom:commentary -->
"""


def _seed_workspace(tmp_path: Path) -> tuple[Workspace, Path, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE, encoding="utf-8")
    page_path = ws.pages / "concepts" / "complementarity.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    return ws, src, page_path


def _seed() -> SeedClaim:
    return SeedClaim(
        entity_id="concept.complementarity",
        entity_type="concept",
        display_name="Complementarity",
        claim_id="c.cmp.1",
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


def test_no_render_persists_claims_but_skips_render(tmp_path: Path) -> None:
    ws, src, page_path = _seed_workspace(tmp_path)
    page_before = page_path.read_text(encoding="utf-8")

    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[_seed()],
        no_render=True,
    )

    assert result.succeeded
    assert result.render_skipped is True
    assert result.pages_rendered == []
    # Claim still persisted normally.
    assert [c.claim_id for c in result.claims_created] == ["c.cmp.1"]
    store = ClaimStore(ws)
    entity = store.load_entity("concept.complementarity")
    assertion = entity.find_assertion("c.cmp.1")
    assert assertion is not None
    assert assertion.verification_status == "verified"

    # The on-disk page bytes are unchanged.
    assert page_path.read_text(encoding="utf-8") == page_before
    # And the claim-block region still contains the original placeholder,
    # not the rendered claim text.
    parsed = parse_page(page_before)
    assert "ORIGINAL_CLAIM_BLOCK_PLACEHOLDER" in parsed.claim_block_inner
    assert "claim:c.cmp.1" not in parsed.claim_block_inner


def test_default_still_renders(tmp_path: Path) -> None:
    """Same setup, no flag: default behavior still renders the page."""
    ws, src, page_path = _seed_workspace(tmp_path)

    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[_seed()],
    )

    assert result.succeeded
    assert result.render_skipped is False
    assert result.pages_rendered, "default behavior must still render"
    parsed = parse_page(page_path.read_text(encoding="utf-8"))
    assert "claim:c.cmp.1" in parsed.claim_block_inner
    assert "ORIGINAL_CLAIM_BLOCK_PLACEHOLDER" not in parsed.claim_block_inner


def test_cli_passes_no_render_flag_through(tmp_path: Path) -> None:
    """`llloom ingest <path> --no-render` reaches the operation and the
    serialized result reports ``render_skipped: true``.

    This exercise drives :func:`llloom.cli.main` directly with crafted
    argv. It does not depend on the seed-claim path because the CLI
    has no surface for that; the model-backed path is exercised via
    the default :class:`NullModel`, which produces zero candidates
    and therefore zero rendered pages with or without the flag. The
    decisive assertion is the ``render_skipped`` flag in the
    serialized JSON, which proves the CLI plumbed the flag through.
    """
    ws, src, _ = _seed_workspace(tmp_path)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(
            [
                "--root",
                str(ws.root),
                "ingest",
                str(src),
                "--source-id",
                "src.article",
                "--source-class",
                "markdown_prose",
                "--no-render",
            ]
        )
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["render_skipped"] is True
    assert payload["pages_rendered"] == []
    # NullModel produces no claims; the flag's effect is solely on the
    # render step.
    assert payload["claims_created"] == []
