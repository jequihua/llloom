"""Unit tests for the verifier mismatch diagnostic.

Hash contract is unchanged: any mismatch still fails. The diagnostic
adds structured context (source id, locator type, both hashes,
bounded previews) without weakening the hash check.
"""

from __future__ import annotations

import pytest

from llloom.claims.models import Assertion, Evidence, Locator
from llloom.claims.verifier import (
    VerifierMismatch,
    compute_excerpt_hash,
    preview_excerpt,
    verify_assertion,
    verify_evidence,
)


SOURCE = """\
# Article

## Methods

A canonical sentence in paragraph one. Another sentence here.
"""


def _good_evidence() -> Evidence:
    locator = Locator(
        locator_type="markdown_prose_v1",
        heading_path=["Methods"],
        paragraph_index=1,
        sentence_start=1,
        sentence_end=1,
    )
    excerpt = "A canonical sentence in paragraph one."
    return Evidence(
        source_id="src.article",
        locator=locator,
        excerpt_hash=compute_excerpt_hash(excerpt, locator.locator_type),
        excerpt=excerpt,
    )


# ---- preview helper -----------------------------------------------------


def test_preview_excerpt_collapses_whitespace() -> None:
    assert preview_excerpt("  foo\n bar  ") == "foo bar"


def test_preview_excerpt_truncates_with_ellipsis() -> None:
    long = "x" * 500
    out = preview_excerpt(long, max_chars=50)
    assert out is not None
    assert len(out) == 50
    assert out.endswith("...")


def test_preview_excerpt_none_passthrough() -> None:
    assert preview_excerpt(None) is None


def test_preview_excerpt_rejects_tiny_max_chars() -> None:
    """The ellipsis marker is three characters, so any ``max_chars``
    below 3 cannot satisfy the documented bound contract: the
    truncation branch ``flat[:max_chars - 3] + "..."`` would itself
    exceed ``max_chars``. The helper must refuse such values up
    front rather than silently violate the bound.

    The check applies even when ``text is None`` (the precondition
    is on the bound, not on whether truncation will actually run),
    so the contract is uniform across every call shape.
    """
    for bad in (0, 1, 2):
        with pytest.raises(ValueError) as exc:
            preview_excerpt("anything", max_chars=bad)
        assert "max_chars must be at least 3" in str(exc.value)
    # Boundary still works: ``max_chars=3`` collapses non-empty text
    # to ``...`` for inputs that do not fit, while ``None`` still
    # round-trips.
    assert preview_excerpt("hello world", max_chars=3) == "..."
    assert preview_excerpt(None, max_chars=3) is None


# ---- structured mismatch on a bare Evidence -----------------------------


def test_verify_evidence_hash_mismatch_emits_structured_diagnostic() -> None:
    ev = _good_evidence()
    bad = Evidence(
        source_id=ev.source_id,
        locator=ev.locator,
        excerpt_hash="sha256:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        excerpt="A previously stored excerpt that is now stale.",
    )
    result = verify_evidence(bad, SOURCE)
    assert not result.passed
    assert result.mismatches, "expected one structured mismatch"
    m = result.mismatches[0]
    assert isinstance(m, VerifierMismatch)
    assert m.claim_id is None  # not called from an assertion context
    assert m.source_id == "src.article"
    assert m.locator_type == "markdown_prose_v1"
    assert m.stored_hash == bad.excerpt_hash
    assert m.computed_hash != bad.excerpt_hash
    assert m.computed_hash.startswith("sha256:")
    # Current preview must be bounded and reflect the actual current span.
    assert m.current_preview
    assert len(m.current_preview) <= 120
    assert "canonical sentence" in m.current_preview
    # Stored preview comes from the evidence's verbatim excerpt field.
    assert m.stored_preview == "A previously stored excerpt that is now stale."
    # The textual note ends up in result.notes; sanity check it carries the
    # same fields a human/agent would scan for.
    assert any("source_id=src.article" in n for n in result.notes)
    assert any("locator_type=markdown_prose_v1" in n for n in result.notes)


def test_verify_evidence_match_emits_no_mismatch() -> None:
    """Regression: matching hash still verifies cleanly with no diagnostic."""
    result = verify_evidence(_good_evidence(), SOURCE)
    assert result.passed
    assert result.mismatches == []
    assert result.notes == []


# ---- structured mismatch propagation through verify_assertion -----------


def test_verify_assertion_propagates_mismatch_with_claim_id() -> None:
    ev = _good_evidence()
    bad = Evidence(
        source_id=ev.source_id,
        locator=ev.locator,
        excerpt_hash="sha256:" + "0" * 64,
        excerpt=None,  # exercises the None stored_preview path
    )
    assertion = Assertion(
        claim_id="c.test.1",
        subject_id="concept.test",
        claim_kind="fact",
        claim_text="t",
        evidence=[bad],
    )
    result = verify_assertion(assertion, {"src.article": SOURCE})
    assert not result.passed
    assert len(result.mismatches) == 1
    m = result.mismatches[0]
    assert m.claim_id == "c.test.1"
    assert m.stored_preview is None
    # Notes are prefixed with the claim id for log readability.
    assert any("claim c.test.1" in n for n in result.notes)


def test_mismatch_preview_is_bounded_against_huge_source() -> None:
    """The preview must not expose unbounded source text even when the
    resolved span itself is large."""
    big_paragraph = " ".join(f"word{i}" for i in range(500))
    big_source = (
        "# T\n\n## Methods\n\n"
        + big_paragraph
        + ".\n"
    )
    locator = Locator(
        locator_type="markdown_prose_v1",
        heading_path=["Methods"],
        paragraph_index=1,
        sentence_start=1,
        sentence_end=1,
    )
    bad = Evidence(
        source_id="src.big",
        locator=locator,
        excerpt_hash="sha256:" + "1" * 64,
    )
    result = verify_evidence(bad, big_source)
    assert not result.passed
    m = result.mismatches[0]
    assert len(m.current_preview) <= 120
    # The full huge paragraph must not appear in any visible text.
    assert "word499" not in m.current_preview
    assert all("word499" not in n for n in result.notes)
