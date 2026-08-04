"""Contract tests for code-backed `claim_extract_and_view_render`.

Six scenarios:

1. a declaration-level code claim under view-render persists AND
   renders the target page
2. an attached-comment code claim under view-render persists AND
   renders the target page
3. commentary on the rendered code-backed page survives byte-for-byte
4. an arbitrary body-span code claim under view-render is refused with
   no partial claim or page writes
5. a detached comment code claim under view-render is refused with no
   partial claim or page writes
6. invocation log persists once and is summary-only on the
   render-enabled code-backed path
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from llloom.claims.store import ClaimStore
from llloom.llm.harness import LLMInvoke
from llloom.ops.ingest import ingest
from llloom.pages.regions import parse_page
from llloom.state.journal import OperationJournal
from llloom.structured.extract import StructureItem, StructureReport
from llloom.workspace.layout import Workspace


_PY_SOURCE_COMMENT_THEN_CLASS = (
    "# Persistent storage abstraction\n"
    "# for the example domain.\n"
    "class Store:\n"
    "    def save(self, item):\n"
    "        return item\n"
)


_PY_SOURCE_DETACHED_COMMENT = (
    "# detached comment line\n"
    "\n"
    "class Store:\n"
    "    def save(self, item):\n"
    "        return item\n"
)


_PAGE_TEMPLATE = """\
---
page_id: code/store
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.code.store -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.code.store owner=human -->

Human commentary that must survive code-backed rerender.

<!-- /llloom:commentary -->
"""


_RAW_PATH = "raw/sources/store.py"


def _wire_view_render_schema(ws: Workspace) -> None:
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  code:\n"
        "    locator: code_v1\n",
        encoding="utf-8",
    )
    (ws.schema / "ingest_policies.yaml").write_text(
        "policies:\n"
        "  markdown_prose: claim_extract_and_view_render\n"
        "  code: claim_extract_and_view_render\n"
        "defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )


def _seed_view_render(
    tmp_path: Path, *, body: str, with_page: bool = True
) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    _wire_view_render_schema(ws)
    src = ws.raw_sources / "store.py"
    src.write_text(body, encoding="utf-8")
    if with_page:
        page = ws.pages / "code" / "store.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(_PAGE_TEMPLATE, encoding="utf-8")
    return ws, src


def _install_fake_extractor(
    monkeypatch: pytest.MonkeyPatch, *, decl_start_line: int
) -> None:
    """Stub `extract_structure` to surface a single class declaration
    on the requested 1-based line so the contract tests run offline
    without tree-sitter (mirrors the prior code-backed slices)."""
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
                        "path": _RAW_PATH,
                        "start_line": decl_start_line,
                        "start_col": 1,
                        "end_line": decl_start_line,
                        "end_col": 12,
                    },
                )
            ],
        )

    monkeypatch.setattr(ingest_mod, "extract_structure", _fake_extract)


class _FakeModel:
    identifier = "fake-code-view-render-model/v0"

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, prompt: str) -> str:
        _ = prompt
        return self._output


_DECL_OUTPUT_WITH_RENDER_TARGET = """\
claims:
  - entity_id: code.store
    entity_type: concept
    display_name: Store
    claim_id: c.store.decl
    claim_kind: definition
    claim_text: Store is a class.
    locator:
      locator_type: code_v1
      path: raw/sources/store.py
      start_line: 3
      start_col: 1
      end_line: 3
      end_col: 12
    render_target: ["code/store", "claim_block.code.store"]
"""


_COMMENT_OUTPUT_WITH_RENDER_TARGET = """\
claims:
  - entity_id: code.store
    entity_type: concept
    display_name: Store
    claim_id: c.store.cmt
    claim_kind: definition
    claim_text: Persistent storage abstraction for the example domain.
    locator:
      locator_type: code_v1
      path: raw/sources/store.py
      start_line: 1
      start_col: 1
      end_line: 2
      end_col: 25
    render_target: ["code/store", "claim_block.code.store"]
"""


_BODY_SPAN_OUTPUT_WITH_RENDER_TARGET = """\
claims:
  - entity_id: code.store
    entity_type: concept
    display_name: Store
    claim_id: c.store.body
    claim_kind: definition
    claim_text: A fragment of the save body.
    locator:
      locator_type: code_v1
      path: raw/sources/store.py
      start_line: 5
      start_col: 9
      end_line: 5
      end_col: 19
    render_target: ["code/store", "claim_block.code.store"]
"""


_DETACHED_OUTPUT_WITH_RENDER_TARGET = """\
claims:
  - entity_id: code.store
    entity_type: concept
    display_name: Store
    claim_id: c.store.detached
    claim_kind: definition
    claim_text: Not actually attached.
    locator:
      locator_type: code_v1
      path: raw/sources/store.py
      start_line: 1
      start_col: 1
      end_line: 1
      end_col: 23
    render_target: ["code/store", "claim_block.code.store"]
"""


# --- 1. declaration-level code claim under view_render persists + renders ---


def test_declaration_code_view_render_persists_claim_and_renders_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_view_render(tmp_path, body=_PY_SOURCE_COMMENT_THEN_CLASS)
    _install_fake_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_DECL_OUTPUT_WITH_RENDER_TARGET)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.store.decl"]
    rendered = [p for p in result.pages_rendered if p.endswith("store.md")]
    assert rendered, f"expected page rendered, got {result.pages_rendered}"

    page_text = (ws.pages / "code" / "store.md").read_text(encoding="utf-8")
    parsed = parse_page(page_text)
    assert "claim:c.store.decl" in parsed.claim_block_inner
    assert "Store is a class" in parsed.claim_block_inner


# --- 2. attached-comment code claim under view_render persists + renders ---


def test_attached_comment_code_view_render_persists_claim_and_renders_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_view_render(tmp_path, body=_PY_SOURCE_COMMENT_THEN_CLASS)
    _install_fake_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_COMMENT_OUTPUT_WITH_RENDER_TARGET)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.store.cmt"]
    rendered = [p for p in result.pages_rendered if p.endswith("store.md")]
    assert rendered

    page_text = (ws.pages / "code" / "store.md").read_text(encoding="utf-8")
    parsed = parse_page(page_text)
    assert "claim:c.store.cmt" in parsed.claim_block_inner


# --- 3. commentary survives byte-for-byte on the rendered code-backed page ---


def test_commentary_survives_byte_for_byte_on_code_backed_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_view_render(tmp_path, body=_PY_SOURCE_COMMENT_THEN_CLASS)
    _install_fake_extractor(monkeypatch, decl_start_line=3)

    page_path = ws.pages / "code" / "store.md"
    pre_commentary_marker = "Human commentary that must survive code-backed rerender."
    assert pre_commentary_marker in page_path.read_text(encoding="utf-8")

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_DECL_OUTPUT_WITH_RENDER_TARGET)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)

    parsed = parse_page(page_path.read_text(encoding="utf-8"))
    assert pre_commentary_marker in parsed.commentary_inner


# --- 4. body-span code claim under view_render refuses, no partial writes ---


def test_body_span_code_view_render_refuses_with_no_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_view_render(tmp_path, body=_PY_SOURCE_COMMENT_THEN_CLASS)
    _install_fake_extractor(monkeypatch, decl_start_line=3)
    page_path = ws.pages / "code" / "store.md"
    pristine = page_path.read_text(encoding="utf-8")

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_BODY_SPAN_OUTPUT_WITH_RENDER_TARGET)),
    )
    assert not result.succeeded
    assert result.refusal_reason
    assert "attached explanation span" in result.refusal_reason
    assert ClaimStore(ws).list_entity_ids() == []
    # The page on disk is byte-identical to the seeded template.
    assert page_path.read_text(encoding="utf-8") == pristine


# --- 5. detached comment under view_render refuses, no partial writes ------


def test_detached_comment_code_view_render_refuses_with_no_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_view_render(tmp_path, body=_PY_SOURCE_DETACHED_COMMENT)
    # Blank line at line 2; declaration at line 3.
    _install_fake_extractor(monkeypatch, decl_start_line=3)
    page_path = ws.pages / "code" / "store.md"
    pristine = page_path.read_text(encoding="utf-8")

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_DETACHED_OUTPUT_WITH_RENDER_TARGET)),
    )
    assert not result.succeeded
    assert result.refusal_reason
    assert "attached explanation span" in result.refusal_reason
    assert ClaimStore(ws).list_entity_ids() == []
    assert page_path.read_text(encoding="utf-8") == pristine


# --- 6. invocation log persists once and is summary-only on render path ---


def test_invocation_log_summary_only_on_code_backed_view_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_view_render(tmp_path, body=_PY_SOURCE_COMMENT_THEN_CLASS)
    _install_fake_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_DECL_OUTPUT_WITH_RENDER_TARGET)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)

    journal = OperationJournal(ws)
    matching = [e for e in journal.iter_entries() if e.op_id == result.op_id]
    assert len(matching) == 1
    entry = matching[0]
    assert len(entry.invocation_logs) == 1
    log = entry.invocation_logs[0]
    for r in log["read_inputs"]:
        assert set(r.keys()) == {"class", "id", "hash"}
    serialized = json.dumps(log)
    assert "def save" not in serialized
    assert "return item" not in serialized
    assert "Persistent storage abstraction" not in serialized


# ---- 7. code-backed view_render on a .cs source -----------------------


_CS_PAGE_TEMPLATE = """\
---
page_id: code/store_cs
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.code.store_cs -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.code.store_cs owner=human -->

Human commentary that must survive C# code-backed rerender.

<!-- /llloom:commentary -->
"""


_CS_VIEW_RENDER_SOURCE = (
    "// Persistent storage abstraction for the example domain.\n"
    "class Store {\n"
    "    void Save() {}\n"
    "}\n"
)


_CS_VIEW_RENDER_OUTPUT = """\
claims:
  - entity_id: code.store_cs
    entity_type: concept
    display_name: Store
    claim_id: c.store.cs.view
    claim_kind: definition
    claim_text: Store is a C# class.
    locator:
      locator_type: code_v1
      path: raw/sources/Store.cs
      start_line: 2
      start_col: 1
      end_line: 2
      end_col: 13
    render_target: ["code/store_cs", "claim_block.code.store_cs"]
"""


def _install_fake_csharp_extractor_for_view_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest_mod = importlib.import_module("llloom.ops.ingest")

    def _fake_extract(source_text, **kwargs):
        return StructureReport(
            source_id=kwargs["source_id"],
            source_class=kwargs["source_class"],
            locator_type="code_v1",
            content_hash=kwargs["content_hash"],
            language="csharp",
            items=[
                StructureItem(
                    kind="class",
                    name="Store",
                    symbol_path="Store",
                    locator={
                        "locator_type": "code_v1",
                        "path": "raw/sources/Store.cs",
                        "start_line": 2,
                        "start_col": 1,
                        "end_line": 2,
                        "end_col": 13,
                    },
                )
            ],
        )

    monkeypatch.setattr(ingest_mod, "extract_structure", _fake_extract)


def test_csharp_code_backed_view_render_persists_claim_and_renders_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A code-backed `claim_extract_and_view_render` on a `.cs`
    source persists a verified declaration-level `code_v1` claim and
    renders the target page through the existing variant-(B)
    contract, with commentary surviving byte-for-byte."""
    ws = Workspace.init(tmp_path)
    _wire_view_render_schema(ws)
    src = ws.raw_sources / "Store.cs"
    src.write_text(_CS_VIEW_RENDER_SOURCE, encoding="utf-8")
    page = ws.pages / "code" / "store_cs.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(_CS_PAGE_TEMPLATE, encoding="utf-8")

    _install_fake_csharp_extractor_for_view_render(monkeypatch)

    result = ingest(
        ws,
        src,
        source_id="src.cs.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_CS_VIEW_RENDER_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.store.cs.view"]
    rendered = [p for p in result.pages_rendered if p.endswith("store_cs.md")]
    assert rendered, f"expected page rendered, got {result.pages_rendered}"

    parsed = parse_page(page.read_text(encoding="utf-8"))
    assert "claim:c.store.cs.view" in parsed.claim_block_inner
    assert "Human commentary that must survive C# code-backed rerender." in parsed.commentary_inner


# ---- 8. Unity component code-backed view_render -----------------------


_UNITY_PAGE_TEMPLATE = """\
---
page_id: code/player_controller
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.code.player_controller -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.code.player_controller owner=human -->

Human commentary that must survive Unity-component rerender.

<!-- /llloom:commentary -->
"""


_UNITY_VIEW_RENDER_SOURCE = (
    "class PlayerController : MonoBehaviour {\n"
    "    void HandleJump() {}\n"
    "}\n"
)


_UNITY_VIEW_RENDER_OUTPUT = """\
claims:
  - entity_id: code.player_controller
    entity_type: concept
    display_name: PlayerController
    claim_id: c.player.unity.view
    claim_kind: definition
    claim_text: PlayerController is a Unity component (MonoBehaviour subclass).
    locator:
      locator_type: code_v1
      path: raw/sources/PlayerController.cs
      start_line: 1
      start_col: 1
      end_line: 1
      end_col: 40
    render_target: ["code/player_controller", "claim_block.code.player_controller"]
"""


def _install_fake_unity_extractor_for_view_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest_mod = importlib.import_module("llloom.ops.ingest")

    def _fake_extract(source_text, **kwargs):
        return StructureReport(
            source_id=kwargs["source_id"],
            source_class=kwargs["source_class"],
            locator_type="code_v1",
            content_hash=kwargs["content_hash"],
            language="csharp",
            items=[
                StructureItem(
                    kind="unity_component",
                    name="PlayerController",
                    symbol_path="PlayerController",
                    locator={
                        "locator_type": "code_v1",
                        "path": "raw/sources/PlayerController.cs",
                        "start_line": 1,
                        "start_col": 1,
                        "end_line": 1,
                        "end_col": 40,
                    },
                )
            ],
        )

    monkeypatch.setattr(ingest_mod, "extract_structure", _fake_extract)


def test_unity_component_code_backed_view_render_persists_and_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A code-backed `claim_extract_and_view_render` on a `.cs`
    source whose structure surfaces a `unity_component` declaration
    persists a verified `code_v1` claim AND renders the target page
    through the unchanged variant-(B) contract. Commentary survives
    byte-for-byte; no Unity-specific renderer is involved."""
    ws = Workspace.init(tmp_path)
    _wire_view_render_schema(ws)
    src = ws.raw_sources / "PlayerController.cs"
    src.write_text(_UNITY_VIEW_RENDER_SOURCE, encoding="utf-8")
    page = ws.pages / "code" / "player_controller.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(_UNITY_PAGE_TEMPLATE, encoding="utf-8")

    _install_fake_unity_extractor_for_view_render(monkeypatch)

    result = ingest(
        ws,
        src,
        source_id="src.cs.player",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_UNITY_VIEW_RENDER_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.player.unity.view"]
    rendered = [p for p in result.pages_rendered if p.endswith("player_controller.md")]
    assert rendered, f"expected page rendered, got {result.pages_rendered}"

    parsed = parse_page(page.read_text(encoding="utf-8"))
    assert "claim:c.player.unity.view" in parsed.claim_block_inner
    assert "Human commentary that must survive Unity-component rerender." in parsed.commentary_inner
