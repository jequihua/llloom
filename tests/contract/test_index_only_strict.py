"""Contract test: strict ``index_only`` ingest never reaches LLMInvoke.

Frozen rule from
``04_specification/storage_and_state_model.md`` Â§"Strict ``index_only``":

- the source is registered and hashed
- no claims are extracted
- no rendered pages are produced
- no LLM-invoking operation may include the source body in its workspace

The pre-hardening implementation constructed a ``SourceDocument`` and
called ``LLMInvoke.invoke`` *before* checking the policy cutoff. This
test plants a spy harness that fails if it is called at all during an
``index_only`` ingest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.llm.harness import (
    InvocationLog,
    LLMInvoke,
    SourceDocument,
    SchemaDocument,
)
from llloom.ops.ingest import ingest
from llloom.workspace.layout import Workspace


class _FailingHarness(LLMInvoke):
    """Test double: any call fails with the operation kind in the error.

    The hardening pass must not invoke this harness during an
    ``index_only`` ingest.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def invoke(self, **kwargs):  # type: ignore[override]
        self.calls.append(kwargs.get("operation_kind", "?"))
        raise AssertionError(
            f"index_only ingest must not call LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r} with "
            f"{len(kwargs.get('source_documents') or [])} source documents"
        )


def _wire_index_only_class(ws: Workspace) -> None:
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


def test_index_only_ingest_does_not_invoke_llm(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    _wire_index_only_class(ws)
    src = ws.raw_sources / "contract.md"
    src.write_text(
        "# Vendor contract\n\nPayment terms: net-30, 2% within 10 days.\n",
        encoding="utf-8",
    )
    spy = _FailingHarness()
    result = ingest(
        ws,
        src,
        source_id="src.contract",
        source_class="sensitive",
        llm=spy,
    )
    assert result.succeeded
    assert result.policy == "index_only"
    assert result.claims_created == []
    assert result.pages_rendered == []
    # The decisive assertion: the spy harness recorded zero calls.
    assert spy.calls == [], (
        f"index_only ingest invoked LLMInvoke; calls={spy.calls}"
    )


def test_structure_extract_ingest_does_not_invoke_llm(tmp_path: Path) -> None:
    """`structure_extract` writes a deterministic derived report and
    must not pass the source body to LLMInvoke. The spy harness fails
    the test if called."""
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "config.yaml"
    src.write_text("policies:\n  markdown_prose: claim_extract\n", encoding="utf-8")
    spy = _FailingHarness()
    result = ingest(
        ws,
        src,
        source_id="src.config",
        source_class="structured_yaml",
        llm=spy,
    )
    assert result.succeeded
    assert result.policy == "structure_extract"
    assert spy.calls == []
    assert result.structure_reports == ["state/structure/src.config.yaml"]
    assert ws.structure_report_path("src.config").is_file()


def test_claim_extract_ingest_does_invoke_llm(tmp_path: Path) -> None:
    """Symmetric positive control: claim_extract_and_view_render still
    routes through LLMInvoke (with NullModel) so the audit log persists."""
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "doc.md"
    src.write_text("# doc\n\n## body\n\nSentence one.\n", encoding="utf-8")
    seen: list[str] = []
    real = LLMInvoke()

    def _wrapped(**kwargs):
        seen.append(kwargs["operation_kind"])
        return LLMInvoke.invoke(real, **kwargs)

    real.invoke = _wrapped  # type: ignore[assignment]
    ingest(ws, src, source_id="src.doc", source_class="markdown_prose", llm=real)
    assert seen == ["ingest"]

