"""Contract tests for metadata-only structure context on
``claim_extract`` ingest of a narrative source.

Persisted claims must still ground in the narrative source; providers
must still refuse ``code_v1`` on the ``claim_extract`` path; missing
or stale requested structure reports must refuse cleanly; the
``index_only`` and ``structure_extract`` cutoffs remain intact even
when ``--structure-source`` is supplied; and the invocation log
remains summary-only.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml as yaml_mod

from llloom.claims.store import ClaimStore
from llloom.cli import main as cli_main
from llloom.llm.harness import LLMInvoke, StructureItemContext
from llloom.ops.ingest import ingest
from llloom.state.journal import OperationJournal
from llloom.workspace.layout import Workspace


# ---- shared fixtures ---------------------------------------------------


ARTICLE = """\
# Article

## Methods

Complementarity prioritizes sites that add features not already represented in the selected set. It is widely used.

A second paragraph that mentions diversity but not the central concept.
"""


YAML_SOURCE = (
    "policies:\n"
    "  markdown_prose: claim_extract_and_view_render\n"
    "  legal_act: claim_extract\n"
    "defaults:\n"
    "  unknown: deny\n"
)


GOOD_OUTPUT = """\
claims:
  - entity_id: concept.complementarity
    entity_type: concept
    display_name: Complementarity
    claim_id: c.cmp.1
    claim_kind: definition
    claim_text: |-
      Complementarity prioritizes sites that add features not already
      represented in the selected set.
    locator:
      locator_type: markdown_prose_v1
      heading_path: ["Methods"]
      paragraph_index: 1
      sentence_start: 1
      sentence_end: 1
"""


GOOD_OUTPUT_WITH_RENDER_TARGET = """\
claims:
  - entity_id: concept.complementarity
    entity_type: concept
    display_name: Complementarity
    claim_id: c.cmp.1
    claim_kind: definition
    claim_text: |-
      Complementarity prioritizes sites that add features not already
      represented in the selected set.
    locator:
      locator_type: markdown_prose_v1
      heading_path: ["Methods"]
      paragraph_index: 1
      sentence_start: 1
      sentence_end: 1
    render_target: ["concept/complementarity", "claim_block.concept.complementarity"]
"""


CODE_LOCATOR_OUTPUT = """\
claims:
  - entity_id: concept.complementarity
    entity_type: concept
    display_name: Complementarity
    claim_id: c.cmp.bad
    claim_kind: definition
    claim_text: A bogus claim.
    locator:
      locator_type: code_v1
      path: raw/sources/policies.yaml
      start_line: 1
      start_col: 1
      end_line: 1
      end_col: 8
"""


class _FakeModel:
    identifier = "fake-structure-context-model/v0"

    def __init__(self, output: str) -> None:
        self._output = output
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._output


PAGE_TEMPLATE = """\
---
page_id: concept/complementarity
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.concept.complementarity -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.complementarity owner=human -->

Human commentary that must survive rerender.

<!-- /llloom:commentary -->
"""


def _seed(tmp_path: Path, *, with_page: bool = False) -> tuple[Workspace, Path, Path]:
    """Seed a workspace with both a YAML structure source and the
    narrative article markdown. Returns ``(ws, article_path, yaml_path)``."""
    ws = Workspace.init(tmp_path)
    article = ws.raw_sources / "article.md"
    article.write_text(ARTICLE, encoding="utf-8")
    policies = ws.raw_sources / "policies.yaml"
    policies.write_text(YAML_SOURCE, encoding="utf-8")
    if with_page:
        page = ws.pages / "concepts" / "complementarity.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(PAGE_TEMPLATE, encoding="utf-8")
    return ws, article, policies


def _ingest_structure(ws: Workspace, path: Path, *, source_id: str) -> None:
    result = ingest(ws, path, source_id=source_id, source_class="structured_yaml")
    assert result.succeeded, result.refusal_reason


# --- 1. claim_extract w/ structure context persists narrative-grounded claim ---


def test_claim_extract_with_structure_context_passes_metadata_and_persists_claim(
    tmp_path: Path,
) -> None:
    ws, article, policies = _seed(tmp_path)
    _ingest_structure(ws, policies, source_id="src.policies")

    fake = _FakeModel(GOOD_OUTPUT)
    result = ingest(
        ws,
        article,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=fake),
        structure_source_ids=["src.policies"],
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.cmp.1"]

    # Structure metadata reached the model prompt.
    assert any(
        "## structure_item src.policies [yaml]" in p for p in fake.prompts
    )
    # The claim is grounded in the article markdown, NOT in the YAML source.
    store = ClaimStore(ws)
    entity = store.load_entity("concept.complementarity")
    assertion = entity.find_assertion("c.cmp.1")
    assert assertion is not None
    assert assertion.evidence[0].source_id == "src.article"
    assert assertion.evidence[0].excerpt is not None


# --- 2. claim_extract_and_view_render with structure context renders pages -----


def test_claim_extract_and_view_render_with_structure_context_renders(
    tmp_path: Path,
) -> None:
    ws, article, policies = _seed(tmp_path, with_page=True)
    _ingest_structure(ws, policies, source_id="src.policies")

    fake = _FakeModel(GOOD_OUTPUT_WITH_RENDER_TARGET)
    result = ingest(
        ws,
        article,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=fake),
        structure_source_ids=["src.policies"],
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    rendered = [p for p in result.pages_rendered if p.endswith("complementarity.md")]
    assert rendered, f"expected page rendered, got {result.pages_rendered}"


# --- 3. provider may not persist code_v1 claims via this path -----------------


def test_code_v1_claim_from_provider_is_refused_no_partial_writes(
    tmp_path: Path,
) -> None:
    """A provider that returns a ``code_v1`` locator for a narrative
    ingest must be refused: the locator cannot resolve against the
    article text, the verifier refuses, and the whole batch is
    refused with no partial writes."""
    ws, article, policies = _seed(tmp_path)
    _ingest_structure(ws, policies, source_id="src.policies")

    fake = _FakeModel(CODE_LOCATOR_OUTPUT)
    result = ingest(
        ws,
        article,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=fake),
        structure_source_ids=["src.policies"],
    )
    assert not result.succeeded
    assert result.refusal_reason
    # No entity got persisted; the only registered claim store after
    # this call must be empty.
    assert ClaimStore(ws).list_entity_ids() == []


# --- 4. missing structure report refuses cleanly ------------------------------


def test_missing_structure_report_refuses_cleanly(tmp_path: Path) -> None:
    """Requested structure source exists in the registry but has no
    on-disk report yet. Caller asked for context explicitly, so this
    is a batch-atomic refusal — silent omission would mislead them."""
    ws, article, policies = _seed(tmp_path)
    _ingest_structure(ws, policies, source_id="src.policies")
    # Delete the report after registration.
    ws.structure_report_path("src.policies").unlink()

    fake = _FakeModel(GOOD_OUTPUT)
    result = ingest(
        ws,
        article,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=fake),
        structure_source_ids=["src.policies"],
    )
    assert not result.succeeded
    assert result.refusal_reason is not None
    assert "no report" in result.refusal_reason
    assert ClaimStore(ws).list_entity_ids() == []


# --- 5. stale structure report refuses cleanly --------------------------------


def test_stale_structure_report_content_hash_refuses(tmp_path: Path) -> None:
    """Report exists but its ``content_hash`` no longer matches the
    registry's hash. Refuse — re-ingest with structure_extract."""
    ws, article, policies = _seed(tmp_path)
    _ingest_structure(ws, policies, source_id="src.policies")
    report_path = ws.structure_report_path("src.policies")
    loaded = yaml_mod.safe_load(report_path.read_text(encoding="utf-8"))
    loaded["content_hash"] = "sha256:" + "0" * 64
    report_path.write_text(
        yaml_mod.safe_dump(loaded, sort_keys=False), encoding="utf-8"
    )

    fake = _FakeModel(GOOD_OUTPUT)
    result = ingest(
        ws,
        article,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=fake),
        structure_source_ids=["src.policies"],
    )
    assert not result.succeeded
    assert result.refusal_reason
    assert "stale" in result.refusal_reason


# --- 6. index_only with --structure-source still does not invoke backend ------


def test_index_only_with_structure_source_skips_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict ``index_only`` cutoff must fire before any harness call,
    even when the caller supplied ``--structure-source``."""
    ws = Workspace.init(tmp_path)
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  sensitive:\n"
        "    locator: markdown_prose_v1\n"
        "  structured_yaml:\n"
        "    locator: code_v1\n",
        encoding="utf-8",
    )
    (ws.schema / "ingest_policies.yaml").write_text(
        "policies:\n"
        "  markdown_prose: claim_extract_and_view_render\n"
        "  sensitive: index_only\n"
        "  structured_yaml: structure_extract\n"
        "defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )
    policies = ws.raw_sources / "policies.yaml"
    policies.write_text(YAML_SOURCE, encoding="utf-8")
    _ingest_structure(ws, policies, source_id="src.policies")

    article = ws.raw_sources / "sensitive.md"
    article.write_text("# Sensitive\n\nSome text.\n", encoding="utf-8")

    def _fail(self, **kwargs):  # noqa: ANN001 - test stub
        raise AssertionError(
            "index_only ingest must not invoke LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r}"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _fail)

    result = ingest(
        ws,
        article,
        source_id="src.sensitive",
        source_class="sensitive",
        structure_source_ids=["src.policies"],
    )
    assert result.succeeded
    assert result.policy == "index_only"


# --- 7. structure_extract with --structure-source stays deterministic --------


def test_structure_extract_with_structure_source_stays_non_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``structure_extract`` ingests must never invoke ``LLMInvoke``,
    even when the caller passed ``--structure-source``. The
    structure_extract policy cutoff in ``ops/ingest.py`` is the
    enforcement point."""
    ws, _, policies = _seed(tmp_path)
    _ingest_structure(ws, policies, source_id="src.policies")

    # A second structured source to be ingested while supplying the
    # first as ``--structure-source``. The cutoff must still fire.
    other = ws.raw_sources / "other.yaml"
    other.write_text("a:\n  b: 1\n", encoding="utf-8")

    def _fail(self, **kwargs):  # noqa: ANN001 - test stub
        raise AssertionError(
            "structure_extract must not invoke LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r}"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _fail)

    result = ingest(
        ws,
        other,
        source_id="src.other",
        source_class="structured_yaml",
        structure_source_ids=["src.policies"],
    )
    assert result.succeeded
    assert result.policy == "structure_extract"
    assert result.structure_reports == ["state/structure/src.other.yaml"]


# --- 8. invocation log persists exactly once and is summary-only -------------


def test_invocation_log_persists_once_with_summary_only_structure_entries(
    tmp_path: Path,
) -> None:
    """The journal entry must record exactly one invocation log, and
    the structure-item read_inputs entries must contain class/id/hash
    only — never any raw scalar value, comment, or code body from the
    source YAML."""
    ws, article, policies = _seed(tmp_path)
    # Plant recognizable strings the structure source's scalar value and
    # a comment-derived path would carry. They must not appear in the
    # invocation log.
    poisoned = ws.raw_sources / "poisoned.yaml"
    poisoned.write_text(
        "# POISON_COMMENT_marker_xyz\n"
        "policies:\n"
        "  one: POISON_VALUE_marker_abc\n",
        encoding="utf-8",
    )
    _ingest_structure(ws, poisoned, source_id="src.poisoned")

    fake = _FakeModel(GOOD_OUTPUT)
    result = ingest(
        ws,
        article,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=fake),
        structure_source_ids=["src.poisoned"],
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)

    journal = OperationJournal(ws)
    entries = list(journal.iter_entries())
    ingest_entries = [e for e in entries if e.op_id == result.op_id]
    assert len(ingest_entries) == 1
    entry = ingest_entries[0]
    assert len(entry.invocation_logs) == 1
    log = entry.invocation_logs[0]
    structure_summaries = [
        r for r in log["read_inputs"] if r["class"] == "StructureItemContext"
    ]
    assert structure_summaries, log["read_inputs"]
    serialized = json.dumps(log)
    assert "POISON_VALUE_marker_abc" not in serialized
    assert "POISON_COMMENT_marker_xyz" not in serialized
    # Each structure summary carries class/id/hash only.
    for r in structure_summaries:
        assert set(r.keys()) == {"class", "id", "hash"}
