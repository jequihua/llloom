"""Contract test: ``query`` against ``index_only`` sources never passes
raw source bodies to ``LLMInvoke``.

The first-slice ``query`` is purely deterministic local retrieval; it
does not invoke the harness at all. This test enforces that property
*structurally*: it monkey-patches ``LLMInvoke.invoke`` to fail on any
call. If a future regression reintroduces a model invocation in the
query path, this test fails immediately.

Even if a later slice routes verbatim spans through the harness for
audit parity, the same property must hold for raw bodies — only
bounded ``SourceSpan`` typed inputs are permitted by the operation
matrix in ``04_specification/component_contracts.md`` §LLMInvoke. This
test guards the strongest version of the property.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.llm.harness import LLMInvoke
from llloom.ops.ingest import ingest
from llloom.ops.query import query
from llloom.workspace.layout import Workspace


SOURCE_TEXT = (
    "# Vendor contract\n\n"
    "## Payment terms\n\n"
    "Standard agreements use net-30 terms with a 2% discount if paid "
    "within 10 days of invoice date.\n"
)


def _wire_index_only(ws: Workspace) -> None:
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


def test_index_only_query_does_not_invoke_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace.init(tmp_path)
    _wire_index_only(ws)
    src = ws.raw_sources / "contract.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    ingest(ws, src, source_id="src.contract", source_class="sensitive")

    calls: list[str] = []

    def _fail_invoke(self, **kwargs):  # noqa: ANN001 - test stub
        calls.append(kwargs.get("operation_kind", "?"))
        raise AssertionError(
            "query against index_only sources must not call LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r}"
        )

    # Monkeypatch every LLMInvoke instance's `invoke` method. If query
    # constructs a harness internally and calls it, this raises and the
    # test fails loudly.
    monkeypatch.setattr(LLMInvoke, "invoke", _fail_invoke)

    result = query(ws, question="What is the early-payment discount?")
    assert calls == [], (
        f"query path invoked LLMInvoke during index_only retrieval; calls={calls}"
    )
    # Sanity: the deterministic retrieval path still produced spans.
    assert result.used_verbatim_spans, (
        "monkeypatch removed the only invocation path; spans must still come "
        "from the deterministic retrieval path"
    )
