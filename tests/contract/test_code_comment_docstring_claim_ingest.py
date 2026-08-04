"""Contract tests for comments-and-docstrings claims on code-backed
``claim_extract`` ingest.

Six scenarios:

1. a leading-comment-block `code_v1` claim persists end-to-end with the
   verifier resolving and hashing the exact comment span
2. an arbitrary body-span `code_v1` claim is refused with no partial
   writes (regression coverage for the explanation-only admission)
3. a detached comment span (blank line between the comment block and
   the next declaration) is refused with no partial writes
4. declaration-level direct `code_v1` claims still persist unchanged
   under the new combined admission validator
5. code-backed ``claim_extract_and_view_render`` still refuses early
   and visibly (rendering remains deferred)
6. invocation logs still persist exactly once and remain summary-only
   on the explanation path
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from llloom.claims.store import ClaimStore
from llloom.llm.harness import LLMInvoke
from llloom.ops.ingest import ingest
from llloom.state.journal import OperationJournal
from llloom.structured.extract import StructureItem, StructureReport
from llloom.workspace.layout import Workspace


# --- shared scaffolding -------------------------------------------------


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

_RAW_PATH = "raw/sources/store.py"


def _wire_code_schema(ws: Workspace, *, view_render: bool = False) -> None:
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  code:\n"
        "    locator: code_v1\n",
        encoding="utf-8",
    )
    policy = "claim_extract_and_view_render" if view_render else "claim_extract"
    (ws.schema / "ingest_policies.yaml").write_text(
        f"policies:\n"
        f"  markdown_prose: claim_extract_and_view_render\n"
        f"  code: {policy}\n"
        f"defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )


def _seed_code_source(
    tmp_path: Path, *, body: str, view_render: bool = False
) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    _wire_code_schema(ws, view_render=view_render)
    src = ws.raw_sources / "store.py"
    src.write_text(body, encoding="utf-8")
    return ws, src


def _install_fake_extractor(
    monkeypatch: pytest.MonkeyPatch, *, decl_start_line: int
) -> None:
    """Stub `extract_structure` to surface a single class declaration
    at the requested 1-based line. Mirrors the prior code-claim test
    setup so the contract test runs offline without tree-sitter."""
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
    identifier = "fake-comment-docstring-model/v0"

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, prompt: str) -> str:
        _ = prompt
        return self._output


_LEADING_COMMENT_OUTPUT = """\
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
"""


_BODY_SPAN_OUTPUT = """\
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
"""


_DETACHED_COMMENT_OUTPUT = """\
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
"""


_DECLARATION_OUTPUT = """\
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
"""


# --- 1. attached comment block persists end-to-end ----------------------


def test_attached_comment_claim_persists_verified_code_v1_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_code_source(tmp_path, body=_PY_SOURCE_COMMENT_THEN_CLASS)
    _install_fake_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_LEADING_COMMENT_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.store.cmt"]

    entity = ClaimStore(ws).load_entity("code.store")
    assertion = entity.find_assertion("c.store.cmt")
    assert assertion is not None
    assert assertion.verification_status == "verified"
    assert assertion.evidence[0].locator.locator_type == "code_v1"
    # The exact two-line comment block was resolved and hashed.
    excerpt = assertion.evidence[0].excerpt
    assert excerpt is not None
    assert excerpt.startswith("# Persistent storage abstraction")
    assert excerpt.endswith("# for the example domain.")


# --- 2. arbitrary body span refused -------------------------------------


def test_arbitrary_body_span_is_refused_with_no_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_code_source(tmp_path, body=_PY_SOURCE_COMMENT_THEN_CLASS)
    _install_fake_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_BODY_SPAN_OUTPUT)),
    )
    assert not result.succeeded
    assert result.refusal_reason
    assert "attached explanation span" in result.refusal_reason
    assert ClaimStore(ws).list_entity_ids() == []


# --- 3. detached comment refused ----------------------------------------


def test_detached_comment_is_refused_with_no_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_code_source(tmp_path, body=_PY_SOURCE_DETACHED_COMMENT)
    # Blank line at line 2; declaration is on line 3.
    _install_fake_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_DETACHED_COMMENT_OUTPUT)),
    )
    assert not result.succeeded
    assert result.refusal_reason
    assert "attached explanation span" in result.refusal_reason
    assert ClaimStore(ws).list_entity_ids() == []


# --- 4. declaration-level direct code_v1 still persists -----------------


def test_declaration_level_claim_still_persists_under_combined_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression coverage for the prior slice: the new combined
    admission must still admit declaration spans exactly. A blanket
    rewrite that lost the declaration shape would fail here."""
    ws, src = _seed_code_source(tmp_path, body=_PY_SOURCE_COMMENT_THEN_CLASS)
    _install_fake_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_DECLARATION_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.store.decl"]
    entity = ClaimStore(ws).load_entity("code.store")
    assertion = entity.find_assertion("c.store.decl")
    assert assertion is not None
    assert assertion.evidence[0].excerpt == "class Store:"


# --- 5. code-backed claim_extract_and_view_render still refuses ---------


def test_code_backed_view_render_admits_attached_comment_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The code-backed `claim_extract_and_view_render` early refusal
    has been removed: an attached comment-block claim now flows
    through the same admission pipeline that `claim_extract` used in
    the prior slice and persists successfully. With no
    `render_target` on the model output and no page on disk, the
    render step emits zero pages — the assertion here is that the
    policy cutoff itself no longer fires and the attached-comment
    claim still persists."""
    ws, src = _seed_code_source(
        tmp_path, body=_PY_SOURCE_COMMENT_THEN_CLASS, view_render=True
    )
    _install_fake_extractor(monkeypatch, decl_start_line=3)
    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_LEADING_COMMENT_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.store.cmt"]
    assert result.pages_rendered == []


# --- 6. invocation log persists once and is summary-only ---------------


def test_invocation_log_persists_once_summary_only_on_explanation_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_code_source(tmp_path, body=_PY_SOURCE_COMMENT_THEN_CLASS)
    _install_fake_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_LEADING_COMMENT_OUTPUT)),
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
    # The leading comment text and method-body text never appear in
    # the serialized log (read_inputs carries class / id / hash only).
    serialized = json.dumps(log)
    assert "Persistent storage abstraction" not in serialized
    assert "def save" not in serialized
    assert "return item" not in serialized


# ---- C# coverage on the code-backed claim_extract path -----------------


_CS_RAW_PATH = "raw/sources/Store.cs"


_CS_SOURCE_DOUBLE_SLASH_THEN_CLASS = (
    "// Persistent storage abstraction\n"
    "// for the example domain.\n"
    "class Store {\n"
    "    void Save() {}\n"
    "}\n"
)


_CS_SOURCE_TRIPLE_SLASH_THEN_CLASS = (
    "/// <summary>Stores items for the example domain.</summary>\n"
    "/// <remarks>Used in tests.</remarks>\n"
    "class Store {\n"
    "    void Save() {}\n"
    "}\n"
)


_CS_SOURCE_DETACHED_COMMENT = (
    "// detached comment line\n"
    "\n"
    "class Store {\n"
    "    void Save() {}\n"
    "}\n"
)


def _wire_csharp_schema(ws: Workspace, *, view_render: bool = False) -> None:
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  code:\n"
        "    locator: code_v1\n",
        encoding="utf-8",
    )
    policy = "claim_extract_and_view_render" if view_render else "claim_extract"
    (ws.schema / "ingest_policies.yaml").write_text(
        f"policies:\n"
        f"  markdown_prose: claim_extract_and_view_render\n"
        f"  code: {policy}\n"
        f"defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )


def _seed_csharp_source(
    tmp_path: Path, *, body: str, view_render: bool = False
) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    _wire_csharp_schema(ws, view_render=view_render)
    src = ws.raw_sources / "Store.cs"
    src.write_text(body, encoding="utf-8")
    return ws, src


def _install_fake_csharp_extractor(
    monkeypatch: pytest.MonkeyPatch, *, decl_start_line: int
) -> None:
    """Stub `extract_structure` to surface a single C# class declaration
    on the requested 1-based line. Mirrors the Python-side helper."""
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
                        "path": _CS_RAW_PATH,
                        "start_line": decl_start_line,
                        "start_col": 1,
                        "end_line": decl_start_line,
                        "end_col": 13,
                    },
                )
            ],
        )

    monkeypatch.setattr(ingest_mod, "extract_structure", _fake_extract)


_CS_DECLARATION_OUTPUT = """\
claims:
  - entity_id: code.store
    entity_type: concept
    display_name: Store
    claim_id: c.store.cs.decl
    claim_kind: definition
    claim_text: Store is a C# class.
    locator:
      locator_type: code_v1
      path: raw/sources/Store.cs
      start_line: 3
      start_col: 1
      end_line: 3
      end_col: 13
"""


_CS_DOUBLE_SLASH_OUTPUT = """\
claims:
  - entity_id: code.store
    entity_type: concept
    display_name: Store
    claim_id: c.store.cs.cmt
    claim_kind: definition
    claim_text: Persistent storage abstraction for the example domain.
    locator:
      locator_type: code_v1
      path: raw/sources/Store.cs
      start_line: 1
      start_col: 1
      end_line: 2
      end_col: 26
"""


_CS_TRIPLE_SLASH_OUTPUT = """\
claims:
  - entity_id: code.store
    entity_type: concept
    display_name: Store
    claim_id: c.store.cs.xmldoc
    claim_kind: definition
    claim_text: Stores items for the example domain. Used in tests.
    locator:
      locator_type: code_v1
      path: raw/sources/Store.cs
      start_line: 1
      start_col: 1
      end_line: 2
      end_col: 37
"""
# `_CS_TRIPLE_SLASH_OUTPUT` end_col matches the second `///` line
# length (37 chars: "/// <remarks>Used in tests.</remarks>"). The
# whole-line span enumerator emits `end_col == len(last_line)`, so
# this aligns with the block's last covered line.


_CS_DETACHED_OUTPUT = """\
claims:
  - entity_id: code.store
    entity_type: concept
    display_name: Store
    claim_id: c.store.cs.detached
    claim_kind: definition
    claim_text: Not actually attached.
    locator:
      locator_type: code_v1
      path: raw/sources/Store.cs
      start_line: 1
      start_col: 1
      end_line: 1
      end_col: 24
"""


def test_csharp_declaration_claim_extract_persists_verified_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_csharp_source(tmp_path, body=_CS_SOURCE_DOUBLE_SLASH_THEN_CLASS)
    _install_fake_csharp_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.cs.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_CS_DECLARATION_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.store.cs.decl"]

    entity = ClaimStore(ws).load_entity("code.store")
    assertion = entity.find_assertion("c.store.cs.decl")
    assert assertion is not None
    assert assertion.evidence[0].locator.locator_type == "code_v1"
    assert assertion.evidence[0].excerpt == "class Store {"


def test_csharp_attached_double_slash_claim_extract_persists_verified_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_csharp_source(tmp_path, body=_CS_SOURCE_DOUBLE_SLASH_THEN_CLASS)
    _install_fake_csharp_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.cs.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_CS_DOUBLE_SLASH_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.store.cs.cmt"]

    entity = ClaimStore(ws).load_entity("code.store")
    assertion = entity.find_assertion("c.store.cs.cmt")
    excerpt = assertion.evidence[0].excerpt
    assert excerpt is not None
    assert excerpt.startswith("// Persistent storage abstraction")
    assert excerpt.endswith("// for the example domain.")


def test_csharp_attached_triple_slash_xmldoc_claim_extract_persists_verified_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C# `///` XML-doc lines naturally pass the existing
    contiguous-comment-block rule (they pass `str.startswith("//")`);
    the same attached-explanation contract admits the two-line
    block immediately above the declaration."""
    ws, src = _seed_csharp_source(tmp_path, body=_CS_SOURCE_TRIPLE_SLASH_THEN_CLASS)
    _install_fake_csharp_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.cs.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_CS_TRIPLE_SLASH_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.store.cs.xmldoc"]

    entity = ClaimStore(ws).load_entity("code.store")
    assertion = entity.find_assertion("c.store.cs.xmldoc")
    excerpt = assertion.evidence[0].excerpt
    assert excerpt is not None
    assert "<summary>" in excerpt
    assert "<remarks>" in excerpt


def test_csharp_detached_comment_is_refused_with_no_partial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_csharp_source(tmp_path, body=_CS_SOURCE_DETACHED_COMMENT)
    _install_fake_csharp_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.cs.store",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_CS_DETACHED_OUTPUT)),
    )
    assert not result.succeeded
    assert result.refusal_reason
    assert "attached explanation span" in result.refusal_reason
    assert ClaimStore(ws).list_entity_ids() == []


# ---- Unity bridge v1: code claims over a MonoBehaviour subclass --------


_UNITY_RAW_PATH = "raw/sources/PlayerController.cs"


_UNITY_PY_SOURCE_LEADING_COMMENT = (
    "// Drives the player character.\n"
    "// Reads input and fires jump events.\n"
    "class PlayerController : MonoBehaviour {\n"
    "    void HandleJump() {}\n"
    "}\n"
)


def _install_fake_unity_extractor(
    monkeypatch: pytest.MonkeyPatch, *, decl_start_line: int
) -> None:
    """Stub `extract_structure` so the transient structure walk
    surfaces a single Unity component declaration on the requested
    1-based line. Mirrors `_install_fake_csharp_extractor` but emits
    ``kind="unity_component"``."""
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
                        "path": _UNITY_RAW_PATH,
                        "start_line": decl_start_line,
                        "start_col": 1,
                        "end_line": decl_start_line,
                        "end_col": 40,
                    },
                )
            ],
        )

    monkeypatch.setattr(ingest_mod, "extract_structure", _fake_extract)


def _wire_unity_schema(ws: Workspace, *, view_render: bool = False) -> None:
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  code:\n"
        "    locator: code_v1\n",
        encoding="utf-8",
    )
    policy = "claim_extract_and_view_render" if view_render else "claim_extract"
    (ws.schema / "ingest_policies.yaml").write_text(
        f"policies:\n"
        f"  markdown_prose: claim_extract_and_view_render\n"
        f"  code: {policy}\n"
        f"defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )


def _seed_unity_source(
    tmp_path: Path, *, body: str, view_render: bool = False
) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    _wire_unity_schema(ws, view_render=view_render)
    src = ws.raw_sources / "PlayerController.cs"
    src.write_text(body, encoding="utf-8")
    return ws, src


_UNITY_DECLARATION_OUTPUT = """\
claims:
  - entity_id: code.player_controller
    entity_type: concept
    display_name: PlayerController
    claim_id: c.player.unity.decl
    claim_kind: definition
    claim_text: PlayerController is a Unity component (MonoBehaviour subclass).
    locator:
      locator_type: code_v1
      path: raw/sources/PlayerController.cs
      start_line: 3
      start_col: 1
      end_line: 3
      end_col: 40
"""


_UNITY_LEADING_COMMENT_OUTPUT = """\
claims:
  - entity_id: code.player_controller
    entity_type: concept
    display_name: PlayerController
    claim_id: c.player.unity.cmt
    claim_kind: definition
    claim_text: Drives the player character. Reads input and fires jump events.
    locator:
      locator_type: code_v1
      path: raw/sources/PlayerController.cs
      start_line: 1
      start_col: 1
      end_line: 2
      end_col: 37
"""


def test_unity_component_declaration_claim_extract_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declaration-level `code_v1` claim over a Unity component
    persists end-to-end under the existing code-backed
    `claim_extract` contract. The kind re-tag (`unity_component`)
    is a metadata flavor on the structure side; the claim contract
    treats the span identically to a `class` declaration."""
    ws, src = _seed_unity_source(
        tmp_path, body=_UNITY_PY_SOURCE_LEADING_COMMENT
    )
    _install_fake_unity_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.cs.player",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_UNITY_DECLARATION_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.player.unity.decl"]

    entity = ClaimStore(ws).load_entity("code.player_controller")
    assertion = entity.find_assertion("c.player.unity.decl")
    assert assertion is not None
    assert assertion.evidence[0].locator.locator_type == "code_v1"
    assert assertion.evidence[0].excerpt == "class PlayerController : MonoBehaviour {"


def test_unity_component_attached_double_slash_claim_extract_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attached `//` leading-comment-block claim over a Unity
    component class persists under the unchanged attached-explanation
    contract; the comment block is two lines immediately above the
    `class PlayerController : MonoBehaviour {` declaration."""
    ws, src = _seed_unity_source(
        tmp_path, body=_UNITY_PY_SOURCE_LEADING_COMMENT
    )
    _install_fake_unity_extractor(monkeypatch, decl_start_line=3)

    result = ingest(
        ws,
        src,
        source_id="src.cs.player",
        source_class="code",
        llm=LLMInvoke(model=_FakeModel(_UNITY_LEADING_COMMENT_OUTPUT)),
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    assert [c.claim_id for c in result.claims_created] == ["c.player.unity.cmt"]

    entity = ClaimStore(ws).load_entity("code.player_controller")
    assertion = entity.find_assertion("c.player.unity.cmt")
    excerpt = assertion.evidence[0].excerpt
    assert excerpt is not None
    assert excerpt.startswith("// Drives the player character.")
    assert excerpt.endswith("// Reads input and fires jump events.")
