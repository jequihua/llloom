"""Contract: neutral `raw_evidence` starter source class (Slice 083).

Pins the lightweight Option A from the Graphab dev-note
(`feedback/2026-05-24_graphab_llloom_feedback_next_priorities.md`
item P5): a starter source class for unsupported or
intentionally unstructured UTF-8 evidence registered for hash
+ exact deterministic retrieval only.

Three contract assertions:

1. Ingest under ``source_class="raw_evidence"`` resolves to the
   existing ``index_only`` policy, registers the source, and
   never invokes ``LLMInvoke`` (even when a failing harness is
   passed in explicitly).
2. The registered source's class is recorded as
   ``raw_evidence``; no claims, pages, or structure reports are
   produced.
3. Deterministic ``query(...)`` returns the existing
   `index_only` verbatim spans for raw-evidence sources — the
   slice routes through the same path the pre-existing
   ``index_only`` retrieval uses, not through a new helper.

No new locator type, no new ingest policy, no new sidecar, no
binary source path. The locator reuse (`markdown_prose_v1`) is
schema-level compatibility, not a semantic claim about the
source body.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.llm.harness import LLMInvoke
from llloom.ops.ingest import ingest
from llloom.ops.query import query
from llloom.sources.registry import SourceRegistry
from llloom.workspace.layout import Workspace


_RAW_EVIDENCE_FIXTURE = (
    "// Project.java\n"
    "package com.example;\n"
    "\n"
    "public class Project {\n"
    "    private int maxsize = NeutralEvidenceMarkerPhrase42;\n"
    "    public int evaluate() { return 1; }\n"
    "}\n"
)


class _FailingHarness(LLMInvoke):
    """Test double: any call fails, asserting the
    ``raw_evidence`` -> ``index_only`` cutoff returns before any
    ``LLMInvoke.invoke(...)`` call and before any
    ``SourceDocument`` carrying the raw body reaches the harness.
    (The harness *object* may be constructed at the top of
    ``ingest`` — what is load-bearing is that no ``invoke`` call
    fires on the raw-evidence path.)
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def invoke(self, **kwargs):  # type: ignore[override]
        self.calls.append(kwargs.get("operation_kind", "?"))
        raise AssertionError(
            "raw_evidence ingest must not call LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r} with "
            f"{len(kwargs.get('source_documents') or [])} source documents"
        )


# ---------------------------------------------------------------------
# 1. Ingest under source_class="raw_evidence" is strict index_only
# ---------------------------------------------------------------------


def test_raw_evidence_ingest_is_strict_index_only(tmp_path: Path) -> None:
    """The starter schema maps ``raw_evidence`` to ``index_only``.
    Ingest under that class:

    - registers the source with ``source_class="raw_evidence"``;
    - reports ``IngestResult.policy == "index_only"``;
    - creates no claims, no pages, no structure reports;
    - never calls ``LLMInvoke.invoke`` (the failing-harness spy
      records zero calls); the policy cutoff returns before any
      ``SourceDocument`` carrying the raw body reaches the harness.

    Re-uses the pre-existing strict ``index_only`` cutoff —
    Slice 083 adds no new policy and no new cutoff path.
    """
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "Project.java"
    src.write_text(_RAW_EVIDENCE_FIXTURE, encoding="utf-8")

    spy = _FailingHarness()
    result = ingest(
        ws,
        src,
        source_id="src.project.java",
        source_class="raw_evidence",
        llm=spy,
    )

    assert result.succeeded
    assert result.policy == "index_only"
    assert result.claims_created == []
    assert result.pages_rendered == []
    assert result.structure_reports == []
    assert spy.calls == [], (
        f"raw_evidence ingest invoked LLMInvoke; calls={spy.calls}"
    )

    # Registry record carries the raw_evidence class.
    registry = SourceRegistry(ws)
    record = registry.load("src.project.java")
    assert record.source_class == "raw_evidence"
    assert record.content_hash.startswith("sha256:")


# ---------------------------------------------------------------------
# 2. Deterministic query returns verbatim spans for raw_evidence
# ---------------------------------------------------------------------


def test_raw_evidence_supports_deterministic_exact_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered ``raw_evidence`` source participates in the
    pre-existing ``index_only`` deterministic retrieval path. A
    query for a distinctive phrase from the raw evidence returns
    at least one verbatim span citing that source — without ever
    invoking the harness.

    ``LLMInvoke.invoke`` is monkey-patched to raise; if the
    retrieval path silently constructed a harness call, the test
    would fail immediately. The deterministic span retrieval is
    the rule the pre-existing
    ``test_index_only_query_safety.py`` already pins for the
    ``sensitive`` example; this test proves the starter
    ``raw_evidence`` class participates through the same path.
    """
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "Project.java"
    src.write_text(_RAW_EVIDENCE_FIXTURE, encoding="utf-8")
    ingest(
        ws,
        src,
        source_id="src.project.java",
        source_class="raw_evidence",
    )

    calls: list[str] = []

    def _fail_invoke(self, **kwargs):  # noqa: ANN001 - test stub
        calls.append(kwargs.get("operation_kind", "?"))
        raise AssertionError(
            "query against raw_evidence sources must not call LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r}"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _fail_invoke)

    result = query(ws, question="NeutralEvidenceMarkerPhrase42")
    assert calls == [], (
        f"query path invoked LLMInvoke during raw_evidence retrieval; "
        f"calls={calls}"
    )
    assert result.used_verbatim_spans, (
        "raw_evidence must participate in the deterministic verbatim "
        "retrieval path; got no spans"
    )
    span = result.used_verbatim_spans[0]
    assert span.source_id == "src.project.java"
    assert "NeutralEvidenceMarkerPhrase42" in span.excerpt


# ---------------------------------------------------------------------
# 3. Raw evidence is structurally indistinguishable from any other
#    index_only source on the post-ingest surface
# ---------------------------------------------------------------------


def test_raw_evidence_leaves_no_structure_or_claim_artifacts(
    tmp_path: Path,
) -> None:
    """Defense-in-depth: walk the canonical write-bearing
    directories and confirm no new page, claim YAML, or
    structure-report file is created by the raw-evidence ingest
    beyond the source registry. The slice's "no claims, no pages,
    no structure reports" promise is structurally pinned by
    snapshotting the set of file names under those directories
    before and after the ingest (file-name equality, not content
    hashing — sufficient for the "no new file" promise this test
    proves).
    """
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "Project.java"
    src.write_text(_RAW_EVIDENCE_FIXTURE, encoding="utf-8")

    pages_before = sorted(p.name for p in ws.pages.rglob("*.md"))
    claims_before = sorted(p.name for p in ws.claims_entities.glob("*.yaml"))
    structure_before = sorted(
        p.name for p in ws.state_structure.rglob("*.yaml")
    ) if ws.state_structure.is_dir() else []

    ingest(
        ws,
        src,
        source_id="src.project.java",
        source_class="raw_evidence",
    )

    pages_after = sorted(p.name for p in ws.pages.rglob("*.md"))
    claims_after = sorted(p.name for p in ws.claims_entities.glob("*.yaml"))
    structure_after = sorted(
        p.name for p in ws.state_structure.rglob("*.yaml")
    ) if ws.state_structure.is_dir() else []

    # The source registry is the only directory that legitimately
    # grew during the ingest — no new page, claim YAML, or
    # structure-report file may appear under pages/, claims/entities/,
    # or state/structure/.
    assert pages_after == pages_before, (
        f"raw_evidence ingest wrote a page: "
        f"before={pages_before} after={pages_after}"
    )
    assert claims_after == claims_before, (
        f"raw_evidence ingest wrote a claim: "
        f"before={claims_before} after={claims_after}"
    )
    assert structure_after == structure_before, (
        f"raw_evidence ingest wrote a structure report: "
        f"before={structure_before} after={structure_after}"
    )

    # The source registry now has exactly one record under the new
    # raw_evidence class.
    registry_ids = SourceRegistry(ws).list_ids()
    assert registry_ids == ["src.project.java"]
