"""Contract tests for the LLMInvoke read-class allow-list per operation.

Fills the P1 review gap: prior to this pass the harness only enforced
write kinds and ``ClaimBlockRegion`` on ingest. The frozen matrix in
``04_specification/component_contracts.md`` Â§LLMInvoke restricts which
typed input classes may appear for each operation. These tests prove
those refusals are enforced, not merely conventional.
"""

from __future__ import annotations

import pytest

from llloom.llm.harness import (
    ALLOWED_READ_CLASSES,
    ClaimBlockRegion,
    ClaimRecord,
    HarnessRefusal,
    LLMInvoke,
    SchemaDocument,
    SourceDocument,
    SourceSpan,
)


@pytest.fixture
def harness() -> LLMInvoke:
    return LLMInvoke()


# ---- ingest -------------------------------------------------------------


def test_ingest_allows_source_claim_schema(harness: LLMInvoke) -> None:
    out, _ = harness.invoke(
        op_id="op.ingest.ok",
        operation_kind="ingest",
        source_documents=[
            SourceDocument(source_id="src.x", source_class="markdown_prose", text="x")
        ],
        claim_records=[
            ClaimRecord(claim_id="c.1", entity_id="e.1", claim_text="t")
        ],
        schema_documents=[SchemaDocument(name="s.yaml", text="{}")],
    )
    assert out == ""  # NullModel


def test_ingest_refuses_source_spans(harness: LLMInvoke) -> None:
    """SourceSpan is the verbatim retrieval class used by query; it
    must not appear during ingest."""
    with pytest.raises(HarnessRefusal) as exc:
        harness.invoke(
            op_id="op.ingest.bad",
            operation_kind="ingest",
            source_spans=[SourceSpan(source_id="src.x", excerpt="leaked")],
        )
    assert "SourceSpan" in str(exc.value)


def test_ingest_refuses_claim_blocks(harness: LLMInvoke) -> None:
    """Already covered elsewhere; re-checked here so the matrix story is
    complete in one file."""
    with pytest.raises(HarnessRefusal):
        harness.invoke(
            op_id="op.ingest.cb",
            operation_kind="ingest",
            claim_blocks=[
                ClaimBlockRegion(page_id="p", block_id="b", rendered_text="x")
            ],
        )


# ---- render -------------------------------------------------------------


def test_render_refuses_raw_source_documents(harness: LLMInvoke) -> None:
    """Renderer must not see raw source bodies."""
    with pytest.raises(HarnessRefusal) as exc:
        harness.invoke(
            op_id="op.render.bad",
            operation_kind="render",
            source_documents=[
                SourceDocument(source_id="src.x", source_class="markdown_prose", text="x")
            ],
        )
    assert "SourceDocument" in str(exc.value)


def test_render_refuses_source_spans(harness: LLMInvoke) -> None:
    with pytest.raises(HarnessRefusal):
        harness.invoke(
            op_id="op.render.spans",
            operation_kind="render",
            source_spans=[SourceSpan(source_id="src.x", excerpt="x")],
        )


def test_render_allows_claims_and_blocks(harness: LLMInvoke) -> None:
    harness.invoke(
        op_id="op.render.ok",
        operation_kind="render",
        claim_records=[ClaimRecord(claim_id="c", entity_id="e", claim_text="t")],
        claim_blocks=[ClaimBlockRegion(page_id="p", block_id="b", rendered_text="r")],
        schema_documents=[SchemaDocument(name="s.yaml", text="{}")],
    )


# ---- query --------------------------------------------------------------


def test_query_refuses_raw_source_documents(harness: LLMInvoke) -> None:
    """Query must never see raw source bodies; index_only sources are
    represented only via deterministic SourceSpan results."""
    with pytest.raises(HarnessRefusal) as exc:
        harness.invoke(
            op_id="op.query.bad",
            operation_kind="query",
            source_documents=[
                SourceDocument(source_id="src.x", source_class="markdown_prose", text="x")
            ],
        )
    assert "SourceDocument" in str(exc.value)


def test_query_allows_spans_and_claims(harness: LLMInvoke) -> None:
    harness.invoke(
        op_id="op.query.ok",
        operation_kind="query",
        claim_records=[ClaimRecord(claim_id="c", entity_id="e", claim_text="t")],
        source_spans=[SourceSpan(source_id="src.x", excerpt="exact")],
        claim_blocks=[ClaimBlockRegion(page_id="p", block_id="b", rendered_text="r")],
        schema_documents=[SchemaDocument(name="s.yaml", text="{}")],
    )


# ---- lint ---------------------------------------------------------------


def test_lint_refuses_raw_source_documents(harness: LLMInvoke) -> None:
    with pytest.raises(HarnessRefusal) as exc:
        harness.invoke(
            op_id="op.lint.bad",
            operation_kind="lint",
            source_documents=[
                SourceDocument(source_id="src.x", source_class="markdown_prose", text="x")
            ],
        )
    assert "SourceDocument" in str(exc.value)


def test_lint_refuses_claim_blocks(harness: LLMInvoke) -> None:
    """Lint operates over structured claims; it does not consume rendered
    page regions."""
    with pytest.raises(HarnessRefusal):
        harness.invoke(
            op_id="op.lint.cb",
            operation_kind="lint",
            claim_blocks=[ClaimBlockRegion(page_id="p", block_id="b", rendered_text="x")],
        )


def test_lint_refuses_source_spans(harness: LLMInvoke) -> None:
    """Source spans are the verbatim retrieval class used by query;
    lint must not see them. Fails if ``ALLOWED_READ_CLASSES["lint"]``
    is accidentally broadened to include ``SourceSpan``."""
    with pytest.raises(HarnessRefusal) as exc:
        harness.invoke(
            op_id="op.lint.spans",
            operation_kind="lint",
            source_spans=[SourceSpan(source_id="src.x", excerpt="exact")],
        )
    assert "SourceSpan" in str(exc.value)


def test_lint_allows_claims_and_schema(harness: LLMInvoke) -> None:
    """Positive control for the lint allow list. Fails if
    ``ALLOWED_READ_CLASSES["lint"]`` is accidentally narrowed below
    ``{ClaimRecord, SchemaDocument}``."""
    out, log = harness.invoke(
        op_id="op.lint.ok",
        operation_kind="lint",
        claim_records=[ClaimRecord(claim_id="c", entity_id="e", claim_text="t")],
        schema_documents=[SchemaDocument(name="s.yaml", text="{}")],
    )
    assert out == ""  # NullModel
    assert log.operation_kind == "lint"


# ---- matrix shape -------------------------------------------------------


def test_matrix_covers_all_operations() -> None:
    expected = {"ingest", "render", "query", "lint"}
    assert set(ALLOWED_READ_CLASSES) == expected

