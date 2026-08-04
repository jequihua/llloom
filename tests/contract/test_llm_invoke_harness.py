"""Contract tests for the typed-input LLMInvoke harness.

These tests assert the frozen enforcement contract:

- no filesystem access
- typed-input classes only
- operation matrix forbids claim-block input on ingest
- excluded classes are unreachable by construction (there is no class
  representing commentary or spine prose; the harness cannot accept
  them even if asked)
- invocation logs record every typed input by class + hash
"""

from __future__ import annotations

import pytest

from llloom.llm.harness import (
    ClaimBlockRegion,
    ClaimRecord,
    HarnessRefusal,
    LLMInvoke,
    NullModel,
    SchemaDocument,
    SourceDocument,
    WriteTarget,
)


def test_invoke_accepts_only_known_operations() -> None:
    harness = LLMInvoke()
    with pytest.raises(HarnessRefusal):
        harness.invoke(op_id="op.x", operation_kind="not_a_real_op")


def test_ingest_refuses_claim_block_inputs() -> None:
    """Claim extraction must not read existing pages."""
    harness = LLMInvoke()
    block = ClaimBlockRegion(
        page_id="concept/x", block_id="block_x", rendered_text="prior output"
    )
    with pytest.raises(HarnessRefusal) as exc:
        harness.invoke(
            op_id="op.y",
            operation_kind="ingest",
            claim_blocks=[block],
        )
    assert "may not receive ClaimBlockRegion" in str(exc.value)


def test_write_targets_are_matrix_enforced() -> None:
    harness = LLMInvoke()
    # `query` allows no write targets at all in the first slice.
    with pytest.raises(HarnessRefusal):
        harness.invoke(
            op_id="op.z",
            operation_kind="query",
            write_targets=[WriteTarget(path="claims/x.yaml", kind="claim_entity")],
        )


def test_invocation_log_records_inputs() -> None:
    harness = LLMInvoke(model=NullModel())
    output, log = harness.invoke(
        op_id="op.log",
        operation_kind="ingest",
        source_documents=[
            SourceDocument(
                source_id="src.demo", source_class="markdown_prose", text="body"
            )
        ],
        schema_documents=[SchemaDocument(name="ingest_policies.yaml", text="{}")],
    )
    assert output == ""  # NullModel emits nothing
    classes = sorted(r["class"] for r in log.read_inputs)
    assert classes == ["SchemaDocument", "SourceDocument"]
    assert log.op_id == "op.log"
    assert log.operation_kind == "ingest"
    assert log.output_hash.startswith("sha256:")


def test_harness_has_no_filesystem_access() -> None:
    """The harness must not expose any file-reading API.

    This is a structural assertion: the public surface of LLMInvoke is
    ``invoke(...)`` only. There is no ``invoke_path`` or ``read_file``
    method. Callers must construct typed inputs themselves.
    """
    harness = LLMInvoke()
    public = {name for name in dir(harness) if not name.startswith("_")}
    assert "invoke" in public
    assert not any(
        name in public
        for name in {"read_file", "open_path", "fetch", "invoke_path", "load"}
    )


def test_excluded_content_classes_are_structurally_absent() -> None:
    """There is no typed input class for commentary or spine prose."""
    import llloom.llm.harness as h

    public_classes = {
        name for name in dir(h)
        if not name.startswith("_") and name[0:1].isupper()
    }
    # These names should not exist at all.
    for forbidden in ("CommentaryRegion", "SpineDocument", "IndexOnlySourceBody"):
        assert forbidden not in public_classes, (
            f"exclusion contract violated: {forbidden} is a typed input class"
        )

