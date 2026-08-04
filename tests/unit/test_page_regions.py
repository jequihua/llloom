"""Unit tests for page region parsing and replacement."""

from __future__ import annotations

import pytest

from llloom.pages.regions import PageParseError, parse_page, replace_claim_block


VALID_PAGE = """\
---
page_id: concept/example
page_class: concept
write_policy: mixed
status: rendered
---

<!-- llloom:claim-block id=claim_block.concept.example -->
## Example

Old rendered content.
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.example owner=human -->

Human commentary body.

<!-- /llloom:commentary -->
"""


def test_parse_page_captures_regions() -> None:
    parsed = parse_page(VALID_PAGE)
    assert parsed.frontmatter["page_id"] == "concept/example"
    assert parsed.claim_block_id == "claim_block.concept.example"
    assert parsed.commentary_id == "commentary.concept.example"
    assert "Old rendered content" in parsed.claim_block_inner
    assert "Human commentary body" in parsed.commentary_inner


def test_replace_claim_block_preserves_commentary_byte_for_byte() -> None:
    parsed = parse_page(VALID_PAGE)
    new_text = replace_claim_block(parsed, "## Example\n\nNew rendered content.")
    reparsed = parse_page(new_text)
    # Commentary region bytes must be unchanged.
    assert reparsed.commentary_inner == parsed.commentary_inner
    assert "New rendered content." in reparsed.claim_block_inner
    assert "Old rendered content" not in reparsed.claim_block_inner


def test_missing_claim_block_fails_hard() -> None:
    text = VALID_PAGE.replace("llloom:claim-block", "llloom:notablock")
    with pytest.raises(PageParseError):
        parse_page(text)


def test_duplicate_commentary_fails_hard() -> None:
    extra = (
        VALID_PAGE
        + "\n<!-- llloom:commentary id=commentary.second owner=human -->\n\n"
        + "extra\n\n<!-- /llloom:commentary -->\n"
    )
    with pytest.raises(PageParseError):
        parse_page(extra)

