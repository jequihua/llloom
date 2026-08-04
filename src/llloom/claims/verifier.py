"""Span verifier.

Given a claim's evidence entries, resolve each locator against the
current raw source, normalize, hash, and compare to the recorded
``excerpt_hash``. Position-only verification is insufficient.

Diagnostics: when a hash check fails, the verifier produces a
structured :class:`VerifierMismatch` alongside the textual note. The
mismatch carries the source id, locator type, both hashes, and
bounded previews of the currently resolved span and the stored
excerpt (when available). Previews are deliberately small and
deterministic so they never serve as a back-channel for unbounded
source content into review surfaces, journals, or test output.

The hash contract itself is unchanged. Mismatches still fail
verification; the diagnostic is human/agent context only.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from llloom.claims.locators import (
    SpanResolutionError,
    normalize_excerpt,
    resolve_span,
)
from llloom.claims.models import Assertion, Evidence


# Bounded preview length. Small enough that previews never accidentally
# carry large source bodies into journals or stdout. Plain ASCII so the
# truncation marker survives any terminal encoding.
_PREVIEW_MAX_CHARS = 120


def preview_excerpt(
    text: str | None, max_chars: int = _PREVIEW_MAX_CHARS
) -> str | None:
    """Return a bounded, whitespace-collapsed preview.

    None in -> None out. Strings longer than ``max_chars`` are
    truncated and marked with ``...``. Whitespace runs are collapsed
    to a single space so the preview fits on one line.

    Public so callers building their own diagnostics (for example
    the ingest pipeline's pre-verifier hash check) can produce
    matching previews without re-implementing the bound rules.

    The function's bound contract is: every non-None return value
    has length ``<= max_chars``. The truncation branch reserves three
    characters for the ``...`` marker, so ``max_chars`` must be at
    least 3. Tighter values would silently violate the bound; the
    function raises ``ValueError`` instead.
    """
    if max_chars < 3:
        raise ValueError("max_chars must be at least 3")
    if text is None:
        return None
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 3] + "..."


# Internal alias retained for the module's own call sites.
_preview = preview_excerpt


@dataclass
class VerifierMismatch:
    """Structured diagnostic for an ``excerpt_hash`` mismatch.

    Attributes:
        claim_id: claim id when verifying within an Assertion;
            None when the caller verified a bare Evidence.
        source_id: the source the evidence cites.
        locator_type: locator class that produced the resolved span.
        stored_hash: the hash recorded on the evidence entry (the
            value the verifier expected to see).
        computed_hash: the hash of the currently resolved + normalized
            span (the value the verifier observed).
        current_preview: bounded preview of the currently resolved
            span. Always present.
        stored_preview: bounded preview of the optional verbatim
            ``excerpt`` recorded on the evidence at write time.
            None if the evidence omitted the excerpt.
    """

    claim_id: str | None
    source_id: str
    locator_type: str
    stored_hash: str
    computed_hash: str
    current_preview: str
    stored_preview: str | None

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        """Single-line human-readable summary used in note strings."""
        prefix = f"claim {self.claim_id} " if self.claim_id else ""
        stored = (
            f" stored_preview={self.stored_preview!r}"
            if self.stored_preview is not None
            else ""
        )
        return (
            f"{prefix}excerpt_hash mismatch: "
            f"source_id={self.source_id} locator_type={self.locator_type} "
            f"stored={self.stored_hash} computed={self.computed_hash} "
            f"current_preview={self.current_preview!r}{stored}"
        )


@dataclass
class VerifierResult:
    passed: bool
    notes: list[str] = field(default_factory=list)
    mismatches: list[VerifierMismatch] = field(default_factory=list)

    def add_failure(
        self,
        note: str,
        *,
        mismatch: VerifierMismatch | None = None,
    ) -> None:
        self.passed = False
        self.notes.append(note)
        if mismatch is not None:
            self.mismatches.append(mismatch)


def resolve_excerpt(evidence: Evidence, source_text: str) -> str:
    """Return the verbatim excerpt for ``evidence`` against ``source_text``."""
    return resolve_span(evidence.locator, source_text)


def compute_excerpt_hash(excerpt: str, locator_type: str) -> str:
    """Return the hex SHA-256 of the normalized excerpt, prefixed sha256:."""
    normalized = normalize_excerpt(excerpt, locator_type)
    h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def verify_evidence(
    evidence: Evidence,
    source_text: str,
    *,
    claim_id: str | None = None,
) -> VerifierResult:
    """Verify one evidence entry against ``source_text``.

    ``claim_id`` is propagated into any :class:`VerifierMismatch` the
    function emits; callers verifying within an :class:`Assertion`
    should pass it.
    """
    result = VerifierResult(passed=True)
    try:
        raw_excerpt = resolve_span(evidence.locator, source_text)
    except SpanResolutionError as exc:
        result.add_failure(f"span resolution failed: {exc}")
        return result
    expected = evidence.excerpt_hash
    actual = compute_excerpt_hash(raw_excerpt, evidence.locator.locator_type)
    if expected != actual:
        mismatch = VerifierMismatch(
            claim_id=claim_id,
            source_id=evidence.source_id,
            locator_type=evidence.locator.locator_type,
            stored_hash=expected,
            computed_hash=actual,
            current_preview=_preview(raw_excerpt) or "",
            stored_preview=_preview(evidence.excerpt),
        )
        result.add_failure(str(mismatch), mismatch=mismatch)
    return result


def verify_assertion(
    assertion: Assertion,
    source_texts: dict[str, str],
) -> VerifierResult:
    """Verify every evidence entry on ``assertion``.

    ``source_texts`` maps ``source_id`` -> current raw text.
    """
    result = VerifierResult(passed=True)
    if not assertion.evidence:
        result.add_failure(f"claim {assertion.claim_id}: no evidence entries")
        return result
    for ev in assertion.evidence:
        if ev.source_id not in source_texts:
            result.add_failure(
                f"claim {assertion.claim_id}: source {ev.source_id!r} unavailable"
            )
            continue
        sub = verify_evidence(
            ev, source_texts[ev.source_id], claim_id=assertion.claim_id
        )
        if not sub.passed:
            for note in sub.notes:
                result.add_failure(
                    f"claim {assertion.claim_id} evidence from {ev.source_id}: {note}"
                )
            # Propagate the structured diagnostics up. The text note
            # already includes a prefix; the mismatch object retains
            # the unprefixed structured fields.
            result.mismatches.extend(sub.mismatches)
    return result
