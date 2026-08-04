"""Integration tests: deterministic verbatim retrieval from
``index_only`` sources at the ``query`` surface.

This slice turns the previously deferred path into working behavior:

- ``query`` now returns ``VerbatimSpan`` results from raw text of
  registered ``index_only`` sources whose policy resolves to
  ``index_only``.
- Strict ``index_only`` is preserved: no claims are extracted at ingest
  and no claim is synthesized from the verbatim spans.
- Canary tokens that fall outside any matched snippet must not appear
  in the answer or the returned spans.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.claims.store import ClaimStore
from llloom.ops.ingest import ingest
from llloom.ops.lint import FIXED_CANARY_TOKEN
from llloom.ops.query import query
from llloom.ops.results import VerbatimSpan
from llloom.workspace.layout import Workspace


def _wire_index_only_class(ws: Workspace) -> None:
    """Add a ``sensitive`` source class mapped to the ``index_only`` policy."""
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  sensitive:\n"
        "    locator: markdown_prose_v1\n",
        encoding="utf-8",
    )
    (ws.schema / "ingest_policies.yaml").write_text(
        "policies:\n"
        "  markdown_prose: claim_extract_and_view_render\n"
        "  sensitive: index_only\n"
        "defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )


CONTRACT_TEXT = (
    "# Vendor contract\n\n"
    "## Payment terms\n\n"
    "Standard agreements use net-30 terms with a 2% discount if paid "
    "within 10 days of invoice date. The discount does not apply to "
    "partial payments, disputed amounts, or amounts subject to "
    "credit-memo offset.\n"
)


def test_index_only_query_returns_verbatim_span(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    _wire_index_only_class(ws)
    src = ws.raw_sources / "contract.md"
    src.write_text(CONTRACT_TEXT, encoding="utf-8")

    ingest_result = ingest(
        ws, src, source_id="src.contract", source_class="sensitive"
    )
    assert ingest_result.succeeded
    assert ingest_result.policy == "index_only"
    assert ingest_result.claims_created == []
    # Strict ingest contract: no entity claim file is created.
    assert ClaimStore(ws).list_entity_ids() == []

    result = query(ws, question="What is the early-payment discount?")

    # At least one span should match (the question contains "discount" and
    # "payment", both substrings present in the source).
    assert result.used_verbatim_spans, (
        "expected verbatim spans from the index_only source"
    )
    span = result.used_verbatim_spans[0]
    assert isinstance(span, VerbatimSpan)
    assert span.source_id == "src.contract"
    assert span.excerpt_hash.startswith("sha256:")
    assert span.char_start >= 0
    assert span.char_end > span.char_start
    # The excerpt must be contiguous source text and contain a query
    # token (case-insensitive).
    assert span.excerpt == CONTRACT_TEXT[span.char_start : span.char_end]
    assert any(t in span.excerpt.lower() for t in {"discount", "payment"})

    # The deterministic distinguishing fact in this fixture is the 2%
    # figure from the Hidden-Flaw motivating example. It must be
    # reachable verbatim.
    assert any("2%" in s.excerpt for s in result.used_verbatim_spans)

    # The answer must cite the spans deterministically and not invent
    # prose beyond the rigid template.
    answer = result.answer
    assert "verbatim span(s) from index_only source(s)" in answer
    assert "src.contract" in answer
    # No authoritative claims expected — none exist.
    assert "authoritative claim" not in answer


def test_index_only_query_skips_retracted_sources(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    _wire_index_only_class(ws)
    src = ws.raw_sources / "contract.md"
    src.write_text(CONTRACT_TEXT, encoding="utf-8")
    ingest(ws, src, source_id="src.contract", source_class="sensitive")

    # Retract the source via the registry directly (no claims to cascade).
    from llloom.sources.registry import SourceRegistry

    SourceRegistry(ws).mark_retracted("src.contract", reason="test")

    result = query(ws, question="What is the early-payment discount?")
    assert result.used_verbatim_spans == [], (
        "retracted index_only sources must not yield verbatim spans"
    )


def test_index_only_query_canary_outside_match_does_not_leak(tmp_path: Path) -> None:
    """A canary token planted in a *different* paragraph of the same
    source must not appear in the returned spans or answer.

    The matched snippet is a bounded ±_SNIPPET_RADIUS window around
    the question token; the canary lives well outside that window.
    """
    ws = Workspace.init(tmp_path)
    _wire_index_only_class(ws)

    # Build a source whose first paragraph contains the canary and
    # whose much later paragraph contains the matchable token.
    canary_paragraph = (
        f"Top of file. Canary present here: {FIXED_CANARY_TOKEN}.\n\n"
    )
    filler = ("\n".join(f"Filler paragraph {i}." for i in range(40))) + "\n\n"
    matched_paragraph = (
        "Section: payment terms.\n\n"
        "Specifically the 2% early-payment discount is available within "
        "ten days of invoice date.\n"
    )
    full_text = canary_paragraph + filler + matched_paragraph

    src = ws.raw_sources / "contract.md"
    src.write_text(full_text, encoding="utf-8")
    ingest(ws, src, source_id="src.contract", source_class="sensitive")

    result = query(ws, question="early-payment discount?")
    assert result.used_verbatim_spans, "expected at least one matching span"

    # No span may contain the canary; the answer string must not either.
    for span in result.used_verbatim_spans:
        assert FIXED_CANARY_TOKEN not in span.excerpt, (
            f"canary leaked into verbatim span at chars "
            f"{span.char_start}:{span.char_end}"
        )
    assert FIXED_CANARY_TOKEN not in result.answer


def test_index_only_query_answer_is_deterministic_template(tmp_path: Path) -> None:
    """The answer string is a rigid textual template, not LLM prose.

    Running the same query twice on the same workspace must produce
    byte-identical answers. This is the no-synthesis assertion: there
    is no nondeterministic generation path in the first slice.
    """
    ws = Workspace.init(tmp_path)
    _wire_index_only_class(ws)
    src = ws.raw_sources / "contract.md"
    src.write_text(CONTRACT_TEXT, encoding="utf-8")
    ingest(ws, src, source_id="src.contract", source_class="sensitive")

    a = query(ws, question="2% discount?")
    b = query(ws, question="2% discount?")
    assert a.answer == b.answer
    assert [s.excerpt_hash for s in a.used_verbatim_spans] == [
        s.excerpt_hash for s in b.used_verbatim_spans
    ]
