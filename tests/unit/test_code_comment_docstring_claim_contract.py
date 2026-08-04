"""Unit tests for the attached-explanation `code_v1` admission contract.

The combined validator in `ops.ingest` admits a candidate iff its
`code_v1` locator matches either a declaration-level structure item
or an attached-explanation span (leading line-comment block above a
declaration, or a Python docstring on the line immediately below a
class / function / async-function declaration). Both shapes are
enumerated deterministically from the current raw source text.
"""

from __future__ import annotations

import importlib

import pytest

from llloom.claims.models import Locator
from llloom.llm.anthropic_backend import _SYSTEM_PROMPT as _ANTHROPIC_PROMPT
from llloom.llm.openai_backend import _INSTRUCTIONS as _OPENAI_PROMPT
from llloom.ops.ingest import (
    CodeClaimContractError,
    SeedClaim,
    _validate_code_v1_declaration_locators,
)
from llloom.structured.extract import StructureItem, StructureReport


ingest_mod = importlib.import_module("llloom.ops.ingest")


_PY_SOURCE_WITH_COMMENT_AND_DOCSTRING = (
    "# leading comment line one\n"
    "# leading comment line two\n"
    "class Store:\n"
    '    """A docstring.\n'
    '    More body.\n'
    '    """\n'
    "    def save(self, item):\n"
    "        return item\n"
)


_RAW_PATH = "raw/sources/store.py"


def _stub_declaration_report(*, decl_start_line: int = 3) -> StructureReport:
    """Single-declaration structure report used by the unit-level
    enumerator. ``decl_start_line=3`` matches the Python source above —
    `class Store:` is line 3."""
    return StructureReport(
        source_id="src.store",
        source_class="code",
        locator_type="code_v1",
        content_hash="sha256:" + "a" * 64,
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


def _seed(**locator_fields) -> SeedClaim:
    return SeedClaim(
        entity_id="code.store",
        entity_type="concept",
        display_name="Store",
        claim_id="c.store.1",
        claim_kind="definition",
        claim_text="Store explanation.",
        locator=Locator(locator_type="code_v1", **locator_fields),
    )


def _install_fake_extractor(
    monkeypatch: pytest.MonkeyPatch, *, report: StructureReport
) -> None:
    monkeypatch.setattr(ingest_mod, "extract_structure", lambda *a, **kw: report)


# --- 1. attached comment block + docstring span are admitted --------------


def test_combined_validator_admits_attached_comment_and_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new combined validator admits both the leading two-line
    comment block (lines 1-2) and the multi-line Python docstring on
    lines 4-6 because each abuts the `class Store:` declaration on
    line 3."""
    _install_fake_extractor(monkeypatch, report=_stub_declaration_report())

    leading_comment = _seed(
        path=_RAW_PATH,
        start_line=1,
        start_col=1,
        end_line=2,
        end_col=len("# leading comment line two"),
    )
    docstring = _seed(
        path=_RAW_PATH,
        start_line=4,
        start_col=1,
        end_line=6,
        end_col=len('    """'),
    )
    ingest_mod._validate_code_v1_claim_locators(
        candidates=[leading_comment, docstring],
        source_text=_PY_SOURCE_WITH_COMMENT_AND_DOCSTRING,
        source_id="src.store",
        source_class="code",
        raw_path=_RAW_PATH,
        content_hash="sha256:" + "a" * 64,
    )


# --- 2. detached comments and arbitrary body spans are refused ------------


def test_combined_validator_refuses_detached_or_fabricated_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A comment block that does not abut a declaration (because a
    blank line sits between it and the next declaration), and an
    arbitrary body-span locator inside the `save` method, both refuse
    with a message naming the attached-explanation contract."""
    detached_source = (
        "# detached comment\n"
        "\n"  # blank line breaks the attachment
        "class Store:\n"
        "    def save(self, item):\n"
        "        return item\n"
    )
    _install_fake_extractor(
        monkeypatch, report=_stub_declaration_report(decl_start_line=3)
    )

    detached = _seed(
        path=_RAW_PATH,
        start_line=1,
        start_col=1,
        end_line=1,
        end_col=len("# detached comment"),
    )
    with pytest.raises(CodeClaimContractError) as exc:
        ingest_mod._validate_code_v1_claim_locators(
            candidates=[detached],
            source_text=detached_source,
            source_id="src.store",
            source_class="code",
            raw_path=_RAW_PATH,
            content_hash="sha256:" + "a" * 64,
        )
    assert "attached explanation span" in str(exc.value)

    body_span = _seed(
        path=_RAW_PATH,
        start_line=5,
        start_col=9,
        end_line=5,
        end_col=19,
    )
    with pytest.raises(CodeClaimContractError):
        ingest_mod._validate_code_v1_claim_locators(
            candidates=[body_span],
            source_text=_PY_SOURCE_WITH_COMMENT_AND_DOCSTRING,
            source_id="src.store",
            source_class="code",
            raw_path=_RAW_PATH,
            content_hash="sha256:" + "a" * 64,
        )


# --- 3. prior declaration-level validator still works unchanged ---------


def test_prior_declaration_only_validator_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prior helper `_validate_code_v1_declaration_locators` is
    untouched: it admits exact declaration matches and refuses any
    other locator (including a valid attached-explanation span — the
    narrower helper does not know about the new contract)."""
    _install_fake_extractor(monkeypatch, report=_stub_declaration_report())

    declaration = _seed(
        path=_RAW_PATH,
        start_line=3,
        start_col=1,
        end_line=3,
        end_col=12,
    )
    _validate_code_v1_declaration_locators(
        candidates=[declaration],
        source_text=_PY_SOURCE_WITH_COMMENT_AND_DOCSTRING,
        source_id="src.store",
        source_class="code",
        raw_path=_RAW_PATH,
        content_hash="sha256:" + "a" * 64,
    )

    leading_comment = _seed(
        path=_RAW_PATH,
        start_line=1,
        start_col=1,
        end_line=2,
        end_col=len("# leading comment line two"),
    )
    with pytest.raises(CodeClaimContractError) as exc:
        _validate_code_v1_declaration_locators(
            candidates=[leading_comment],
            source_text=_PY_SOURCE_WITH_COMMENT_AND_DOCSTRING,
            source_id="src.store",
            source_class="code",
            raw_path=_RAW_PATH,
            content_hash="sha256:" + "a" * 64,
        )
    assert "declaration" in str(exc.value).lower()


# --- 4. provider instructions describe both shapes -----------------------


def test_provider_instructions_describe_declaration_and_explanation() -> None:
    """Both adapter prompts must describe the dual admission contract:
    declaration-level spans AND attached explanation (line-comment
    block + Python docstring). Arbitrary body spans and detached
    comments remain explicitly forbidden."""
    for prompt in (_OPENAI_PROMPT, _ANTHROPIC_PROMPT):
        assert "declaration-level spans" in prompt
        assert "attached explanation spans" in prompt
        # The two concrete shapes named.
        assert "line-comment block immediately above" in prompt
        assert "triple-quoted docstring" in prompt
        # The refusal anchors.
        assert "detached comments" in prompt
        assert "arbitrary code-body spans" in prompt


# ---- C# attached-explanation admission ---------------------------------


_CS_RAW_PATH = "raw/sources/Store.cs"


_CS_SOURCE_DOUBLE_AND_TRIPLE = (
    "// Persistent storage abstraction\n"
    "// for the example domain.\n"
    "/// <summary>Stores items.</summary>\n"
    "/// <remarks>Used in tests.</remarks>\n"
    "class Store {\n"
    "    void Save() {}\n"
    "}\n"
)


def _stub_cs_declaration_report() -> StructureReport:
    """Single C# class declaration at line 5 (immediately after the
    four-line `//` + `///` comment block in
    ``_CS_SOURCE_DOUBLE_AND_TRIPLE``)."""
    return StructureReport(
        source_id="src.store.cs",
        source_class="code",
        locator_type="code_v1",
        content_hash="sha256:" + "b" * 64,
        language="csharp",
        items=[
            StructureItem(
                kind="class",
                name="Store",
                symbol_path="Store",
                locator={
                    "locator_type": "code_v1",
                    "path": _CS_RAW_PATH,
                    "start_line": 5,
                    "start_col": 1,
                    "end_line": 5,
                    "end_col": 14,
                },
            )
        ],
    )


def _cs_seed(**locator_fields) -> SeedClaim:
    return SeedClaim(
        entity_id="code.store",
        entity_type="concept",
        display_name="Store",
        claim_id="c.store.cs",
        claim_kind="definition",
        claim_text="C# attached explanation.",
        locator=Locator(locator_type="code_v1", **locator_fields),
    )


def test_combined_validator_admits_attached_csharp_double_and_triple_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The combined validator admits an attached four-line C# comment
    block that mixes `//` and `///` lines (rows 1-4, immediately above
    the `class Store {` declaration on row 5). Both `//` and `///`
    lines pass `str.startswith("//")` so the existing contiguous
    line-comment-block rule above a declaration captures the entire
    block as one whole-line span."""
    monkeypatch.setattr(
        ingest_mod, "extract_structure", lambda *a, **kw: _stub_cs_declaration_report()
    )
    block = _cs_seed(
        path=_CS_RAW_PATH,
        start_line=1,
        start_col=1,
        end_line=4,
        end_col=len("/// <remarks>Used in tests.</remarks>"),
    )
    ingest_mod._validate_code_v1_claim_locators(
        candidates=[block],
        source_text=_CS_SOURCE_DOUBLE_AND_TRIPLE,
        source_id="src.store.cs",
        source_class="code",
        raw_path=_CS_RAW_PATH,
        content_hash="sha256:" + "b" * 64,
    )


def test_combined_validator_refuses_detached_csharp_comment_and_body_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two C# refusals in one test:

    1. a `//` comment block that does not abut a declaration (a blank
       line sits between the comment block and the `class Store {`
       line) refuses with `"attached explanation span"` in the
       message
    2. an arbitrary body-span locator inside the `Save` method refuses
       with the same message
    """
    detached_source = (
        "// detached comment line\n"
        "\n"  # blank line breaks the attachment
        "class Store {\n"
        "    void Save() {}\n"
        "}\n"
    )
    detached_report = StructureReport(
        source_id="src.store.cs",
        source_class="code",
        locator_type="code_v1",
        content_hash="sha256:" + "c" * 64,
        language="csharp",
        items=[
            StructureItem(
                kind="class",
                name="Store",
                symbol_path="Store",
                locator={
                    "locator_type": "code_v1",
                    "path": _CS_RAW_PATH,
                    "start_line": 3,
                    "start_col": 1,
                    "end_line": 3,
                    "end_col": 14,
                },
            )
        ],
    )

    monkeypatch.setattr(
        ingest_mod, "extract_structure", lambda *a, **kw: detached_report
    )

    detached = _cs_seed(
        path=_CS_RAW_PATH,
        start_line=1,
        start_col=1,
        end_line=1,
        end_col=len("// detached comment line"),
    )
    with pytest.raises(CodeClaimContractError) as exc:
        ingest_mod._validate_code_v1_claim_locators(
            candidates=[detached],
            source_text=detached_source,
            source_id="src.store.cs",
            source_class="code",
            raw_path=_CS_RAW_PATH,
            content_hash="sha256:" + "c" * 64,
        )
    assert "attached explanation span" in str(exc.value)

    body_span = _cs_seed(
        path=_CS_RAW_PATH,
        start_line=4,
        start_col=5,
        end_line=4,
        end_col=14,
    )
    with pytest.raises(CodeClaimContractError):
        ingest_mod._validate_code_v1_claim_locators(
            candidates=[body_span],
            source_text=detached_source,
            source_id="src.store.cs",
            source_class="code",
            raw_path=_CS_RAW_PATH,
            content_hash="sha256:" + "c" * 64,
        )
