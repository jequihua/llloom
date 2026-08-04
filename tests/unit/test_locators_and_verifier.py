"""Unit tests for locator resolution and the span verifier."""

from __future__ import annotations

import pytest

from llloom.claims.locators import (
    SpanResolutionError,
    normalize_excerpt,
    resolve_span,
)
from llloom.claims.models import (
    Assertion,
    Evidence,
    Locator,
)
from llloom.claims.verifier import (
    compute_excerpt_hash,
    verify_assertion,
    verify_evidence,
)


MARKDOWN_PROSE = """\
# Title

Intro paragraph sentence one. Intro paragraph sentence two.

## Methods

Methods paragraph one sentence A. Methods paragraph one sentence B.

Methods paragraph two.
"""


def test_markdown_prose_resolve_sentence_range() -> None:
    loc = Locator(
        locator_type="markdown_prose_v1",
        heading_path=["Methods"],
        paragraph_index=1,
        sentence_start=1,
        sentence_end=1,
    )
    excerpt = resolve_span(loc, MARKDOWN_PROSE)
    assert excerpt == "Methods paragraph one sentence A."


def test_markdown_prose_paragraph_out_of_range() -> None:
    loc = Locator(
        locator_type="markdown_prose_v1",
        heading_path=["Methods"],
        paragraph_index=99,
        sentence_start=1,
        sentence_end=1,
    )
    with pytest.raises(SpanResolutionError) as exc:
        resolve_span(loc, MARKDOWN_PROSE)
    assert "out of range" in str(exc.value)


def test_normalize_collapses_whitespace() -> None:
    assert normalize_excerpt("  foo\n bar  ", "markdown_prose_v1") == "foo bar"


def test_excerpt_hash_is_stable() -> None:
    h1 = compute_excerpt_hash("foo bar", "markdown_prose_v1")
    h2 = compute_excerpt_hash("foo   bar\n", "markdown_prose_v1")
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_verify_evidence_hash_mismatch() -> None:
    excerpt = "Methods paragraph one sentence A."
    correct = compute_excerpt_hash(excerpt, "markdown_prose_v1")
    bad = "sha256:deadbeef"
    ev_good = Evidence(
        source_id="src.t",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        excerpt_hash=correct,
    )
    ev_bad = Evidence(
        source_id="src.t",
        locator=ev_good.locator,
        excerpt_hash=bad,
    )
    assert verify_evidence(ev_good, MARKDOWN_PROSE).passed
    assert not verify_evidence(ev_bad, MARKDOWN_PROSE).passed


def test_verify_assertion_missing_source() -> None:
    assertion = Assertion(
        claim_id="c1",
        subject_id="e1",
        claim_kind="fact",
        claim_text="text",
        evidence=[
            Evidence(
                source_id="src.missing",
                locator=Locator(locator_type="markdown_prose_v1"),
                excerpt_hash="sha256:aa",
            )
        ],
    )
    result = verify_assertion(assertion, source_texts={})
    assert not result.passed
    assert any("unavailable" in n for n in result.notes)


LEGAL_ACT = """\
CALIFORNIA FISH AND GAME CODE

```
2800.  This chapter shall be known as the Act.
```

```
2801.  The Legislature finds:
   (a) The first finding.
   (b) The second finding continues here. Second sentence of clause b.
```
"""


def test_legal_act_clause_resolution() -> None:
    loc = Locator(
        locator_type="legal_act_v1",
        act_title="California Fish and Game Code",
        section_label="Section 2801",
        clause_label="(b)",
        paragraph_index=1,
        sentence_start=1,
        sentence_end=1,
    )
    excerpt = resolve_span(loc, LEGAL_ACT)
    assert excerpt.startswith("(b) The second finding")


def test_code_locator_requires_bounds() -> None:
    """The `code_v1` locator shape became real in the
    structured-source ingest slice. A locator without
    start/end line/col must refuse rather than silently returning an
    empty excerpt."""
    loc = Locator(locator_type="code_v1", path="x.py")
    with pytest.raises(SpanResolutionError) as exc:
        resolve_span(loc, "def foo(): pass\n")
    assert "start_line" in str(exc.value) or "required" in str(exc.value)

