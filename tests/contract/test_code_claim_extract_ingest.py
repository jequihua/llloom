"""Contract tests for direct ``code_v1`` claims on code-backed
``claim_extract`` ingest.

Six scenarios:

1. a code-backed ``claim_extract`` ingest persists a verified
   ``code_v1`` claim grounded in a declaration-level span
2. a provider-emitted narrative locator on the code path is refused
   batch-atomically with no partial writes
3. a provider-emitted ``code_v1`` locator that does NOT match a
   declaration-level structure item is refused batch-atomically
4. a code-backed source class configured with
   ``claim_extract_and_view_render`` refuses clearly and early
5. the invocation log persists exactly once and is summary-only
6. the narrative ``claim_extract`` path still refuses
   provider-emitted ``code_v1``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llloom.claims.store import ClaimStore
from llloom.llm.harness import LLMInvoke
from llloom.ops.ingest import ingest
from llloom.state.journal import OperationJournal
from llloom.structured.extract import StructureItem, StructureReport
from llloom.workspace.layout import Workspace


_PY_SOURCE = (
    "class Store:\n"
    "    def save(self, item):\n"
    "        return item\n"
)


def _wire_code_schema(ws: Workspace, *, view_render: bool = False) -> None:
    """Override the starter schema so a ``.py`` source class
    resolves to ``claim_extract`` (or, for scenario 4,
    ``claim_extract_and_view_render``). The starter mapping pins
    `code -> structure_extract`, which we deliberately bypass for
    this slice."""
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  code:\n"
        "    locator: code_v1\n",
        encoding="utf-8",
    )
    policy = (
        "claim_extract_and_view_render" if view_render else "claim_extract"
    )
    (ws.schema / "ingest_policies.yaml").write_text(
        f"policies:\n"
        f"  markdown_prose: claim_extract_and_view_render\n"
        f"  code: {policy}\n"
        f"defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )


def _seed_code_source(tmp_path: Path, *, view_render: bool = False) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    _wire_code_schema(ws, view_render=view_render)
    src = ws.raw_sources / "store.py"
    src.write_text(_PY_SOURCE, encoding="utf-8")
    return ws, src


def _install_fake_extractor(
    monkeypatch: pytest.MonkeyPatch, *, raw_path: str
) -> None:
    """Stub `extract_structure` so the code-backed contract tests run
    offline without tree-sitter. The single declaration item is the
    class definition span on line 1, columns 1-12."""
    import importlib

    ingest_mod = importlib.import_module("llloom.ops.ingest")

    def _fake_extract(source_text, **kwargs):
        return StructureReport(
            source_id=kwargs["source_id"],
            source_class=kwargs["source_class"],
            locator_type="code_v1",
            content_hash=kwargs["content_hash"],
            language="python",
            items=[
                StructureItem(
                    kind="class",
                    name="Store",
                    symbol_path="Store",
                    locator={
                        "locator_type": "code_v1",
                        "path": raw_path,
                        "start_line": 1,
                        "start_col": 1,
                        "end_line": 1,
                        "end_col": 12,
                    },
                )
            ],
        )

    monkeypatch.setattr(ingest_mod, "extract_structure", _fake_extract)


class _FakeModel:
    identifier = "fake-code-extractor/v0"

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, prompt: str) -> str:
        _ = prompt
        return self._output


_GOOD_CODE_OUTPUT = """\
claims:
  - entity_id: code.store
    entity_type: concept
    display_name: Store
    claim_id: c.store.1
    claim_kind: definition
    claim_text: Store is a class.
    locator:
      locator_type: code_v1
      path: raw/sources/store.py
      start_line: 1
      start_col: 1
      end_line: 1
      end_col: 12
"""


_NARRATIVE_ON_CODE_OUTPUT = """\
claims:
  - entity_id: code.store
    entity_type: concept
    display_name: Store
    claim_id: c.store.narrative
    claim_kind: definition
    claim_text: Store is a class.
    locator:
      locator_type: markdown_prose_v1
      heading_path: [Methods]
      paragraph_index: 1
      sentence_start: 1
      sentence_end: 1
"""


_BODY_SPAN_CODE_OUTPUT = """\
claims:
  - entity_id: code.store
    entity_type: concept
    display_name: Store
    claim_id: c.store.body
    claim_kind: definition
    claim_text: A body fragment.
    locator:
      locator_type: code_v1
      path: raw/sources/store.py
      start_line: 3
      start_col: 9
      end_line: 3
      end_col: 19
"""


# --- 1. happy path: code-backed claim_extract persists a verified claim ---


def test_code_backed_claim_extract_persists_verified_code_v1_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_code_source(tmp_path)
    _install_fake_extractor(monkeypatch, raw_path="raw/sources/store.py")

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_GOOD_CODE_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.store.1"]

    entity = ClaimStore(ws).load_entity("code.store")
    assertion = entity.find_assertion("c.store.1")
    assert assertion is not None
    assert assertion.verification_status == "verified"
    assert assertion.evidence[0].source_id == "src.store"
    assert assertion.evidence[0].excerpt == "class Store:"
    assert assertion.evidence[0].locator.locator_type == "code_v1"


# --- 2. narrative locator on code path is refused, no partial writes -----


def test_narrative_locator_on_code_path_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_code_source(tmp_path)
    _install_fake_extractor(monkeypatch, raw_path="raw/sources/store.py")

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_NARRATIVE_ON_CODE_OUTPUT)),
    )
    assert not result.succeeded
    assert result.refusal_reason and "markdown_prose_v1" in result.refusal_reason
    assert ClaimStore(ws).list_entity_ids() == []


# --- 3. body-span code_v1 (not a declaration item) is refused ------------


def test_body_span_code_v1_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The locator is structurally valid `code_v1` and resolves
    against the raw source — it would even verify. The contract is
    enforced by `_validate_code_v1_declaration_locators` BEFORE the
    verifier runs."""
    ws, src = _seed_code_source(tmp_path)
    _install_fake_extractor(monkeypatch, raw_path="raw/sources/store.py")

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_BODY_SPAN_CODE_OUTPUT)),
    )
    assert not result.succeeded
    assert result.refusal_reason
    assert "declaration" in result.refusal_reason.lower()
    assert ClaimStore(ws).list_entity_ids() == []


# --- 4. code-backed claim_extract_and_view_render refuses clearly --------


def test_code_backed_view_render_is_no_longer_refused_at_policy_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The code-backed `claim_extract_and_view_render` early refusal
    has been removed: the ingest now flows through the same parse /
    validate / verify / persist pipeline as code-backed
    `claim_extract`, and falls through to the render step. With no
    `render_target` on the model output and no page on disk, the
    render step emits zero pages — the assertion here is that the
    policy cutoff itself no longer fires and the claim still
    persists."""
    ws, src = _seed_code_source(tmp_path, view_render=True)
    _install_fake_extractor(monkeypatch, raw_path="raw/sources/store.py")
    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_GOOD_CODE_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.store.1"]
    assert result.pages_rendered == []


# --- 5. invocation log persists exactly once and is summary-only ---------


def test_invocation_log_persists_once_and_summary_only_on_code_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_code_source(tmp_path)
    _install_fake_extractor(monkeypatch, raw_path="raw/sources/store.py")

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_GOOD_CODE_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)

    journal = OperationJournal(ws)
    matching = [e for e in journal.iter_entries() if e.op_id == result.op_id]
    assert len(matching) == 1
    entry = matching[0]
    assert len(entry.invocation_logs) == 1
    log = entry.invocation_logs[0]
    # Class / id / hash only per read input — no raw code body leaks.
    for r in log["read_inputs"]:
        assert set(r.keys()) == {"class", "id", "hash"}
    serialized = json.dumps(log)
    assert "def save" not in serialized
    assert "return item" not in serialized


# --- 6. narrative claim_extract still refuses provider-emitted code_v1 ---


def test_narrative_claim_extract_still_refuses_code_v1(
    tmp_path: Path,
) -> None:
    """Regression: the source-class-aware admission must keep the
    narrative ingest path refusing code_v1, matching the prior slice's
    contract."""
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(
        "# Article\n\n## Methods\n\nSome prose sentence.\n",
        encoding="utf-8",
    )
    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=_FakeModel(_GOOD_CODE_OUTPUT)),
    )
    assert not result.succeeded
    assert result.refusal_reason and "code_v1" in result.refusal_reason
    assert ClaimStore(ws).list_entity_ids() == []
