"""Unit tests for the metadata-only ``StructureItemContext`` typed input.

The harness must accept structure context only on ``ingest`` and refuse
it on render/query/lint. Prompt serialization and invocation-log
summaries carry structure metadata only — no scalar values, comments,
docstrings, or code bodies.
"""

from __future__ import annotations

import pytest

from llloom.llm.harness import (
    ALLOWED_READ_CLASSES,
    ClaimRecord,
    HarnessRefusal,
    LLMInvoke,
    SchemaDocument,
    SourceDocument,
    StructureItemContext,
    _read_inputs_summary,
)


def _item(symbol: str = "Store.save", *, name: str = "save") -> StructureItemContext:
    return StructureItemContext(
        source_id="src.code.store",
        source_class="code",
        language="python",
        kind="method",
        name=name,
        symbol_path=symbol,
        report_path="state/structure/src.code.store.yaml",
    )


def test_structure_item_context_hash_is_deterministic_and_metadata_only() -> None:
    """The content hash is a function of the metadata fields only.

    No body text, no scalar value, no comment, no docstring contributes
    to the hash; equal metadata produces equal hashes, and a metadata
    edit shifts the hash. Fails if scalar/body text ever sneaks into
    the dataclass.
    """
    a = _item("Store.save", name="save")
    b = _item("Store.save", name="save")
    assert a.content_hash == b.content_hash
    assert a.content_hash.startswith("sha256:")
    c = _item("Store.delete", name="delete")
    assert c.content_hash != a.content_hash
    # The dataclass exposes structure metadata only; assert the field
    # set has not silently widened to include code-text bearing keys.
    assert set(a.__dataclass_fields__.keys()) == {
        "source_id",
        "source_class",
        "language",
        "kind",
        "name",
        "symbol_path",
        "report_path",
    }


def test_assemble_prompt_includes_structure_items_in_order() -> None:
    """``_assemble_prompt`` serializes structure-item blocks in input
    order with metadata-only fields. Verifies deterministic prompt
    composition for audit purposes."""
    items = [
        _item("Store.save", name="save"),
        _item("Store.delete", name="delete"),
    ]
    prompt = LLMInvoke._assemble_prompt(
        "ingest", [], [], [], [], [], items
    )
    save_idx = prompt.index("symbol=Store.save")
    delete_idx = prompt.index("symbol=Store.delete")
    assert save_idx < delete_idx
    assert "## structure_item src.code.store [python] kind=method" in prompt
    assert "name: save\nreport: state/structure/src.code.store.yaml" in prompt
    # Prompt is metadata-only: no scalar/body markers leak.
    assert "POISON_VALUE" not in prompt


def test_structure_item_context_allowed_on_ingest_refused_elsewhere() -> None:
    """``StructureItemContext`` belongs in the ingest allow-list and no
    other operation. A render/query/lint invocation carrying structure
    items must refuse."""
    assert "StructureItemContext" in ALLOWED_READ_CLASSES["ingest"]
    for op in ("render", "query", "lint"):
        assert "StructureItemContext" not in ALLOWED_READ_CLASSES[op]

    harness = LLMInvoke()
    out, log = harness.invoke(
        op_id="op.ingest.ok",
        operation_kind="ingest",
        source_documents=[
            SourceDocument(source_id="src.x", source_class="markdown_prose", text="x")
        ],
        schema_documents=[SchemaDocument(name="s.yaml", text="{}")],
        structure_items=[_item()],
    )
    assert out == ""
    assert any(r["class"] == "StructureItemContext" for r in log.read_inputs)

    for op in ("render", "query", "lint"):
        with pytest.raises(HarnessRefusal) as exc:
            harness.invoke(
                op_id=f"op.{op}.bad",
                operation_kind=op,
                structure_items=[_item()],
            )
        assert "StructureItemContext" in str(exc.value)


def test_read_inputs_summary_records_structure_items_metadata_only() -> None:
    """The invocation log summary for a structure item carries class +
    id + content hash only — never raw code text, comments, docstrings,
    or scalar values."""
    item = _item("Store.save", name="save")
    summary = _read_inputs_summary([], [], [], [], [], [item])
    assert summary == [
        {
            "class": "StructureItemContext",
            "id": "src.code.store:Store.save",
            "hash": item.content_hash,
        }
    ]
    # Defensive: no surprise keys could carry scalar/body bytes.
    assert set(summary[0].keys()) == {"class", "id", "hash"}
