"""Unit tests for source-class-aware locator admission and the
declaration-level code_v1 contract.

Covers the four behaviors the slice opens:

1. the strict YAML parser accepts ``code_v1`` only when it is in the
   explicit ``allowed_locator_types`` set
2. the parser still refuses ``code_v1`` when only a narrative locator
   type is allowed
3. ``_validate_code_v1_declaration_locators`` accepts an exact match
   against a structure-item locator and refuses a fabricated span
4. both provider instruction strings are source-class-aware (narrative
   forbidden / code-backed declaration-only) — not a blanket "never
   emit code_v1"
"""

from __future__ import annotations

import pytest

from llloom.claims.models import Locator
from llloom.llm.anthropic_backend import _SYSTEM_PROMPT as _ANTHROPIC_PROMPT
from llloom.llm.openai_backend import _INSTRUCTIONS as _OPENAI_PROMPT
from llloom.llm.output import ModelOutputError, parse_claim_extraction_output
from llloom.ops.ingest import (
    CodeClaimContractError,
    SeedClaim,
    _validate_code_v1_declaration_locators,
)


_PY_SOURCE = (
    "class Store:\n"
    "    def save(self, item):\n"
    "        return item\n"
)


_CODE_OUTPUT = """\
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
      end_line: 3
      end_col: 20
"""


_NARRATIVE_OUTPUT = """\
claims:
  - entity_id: concept.x
    entity_type: concept
    display_name: X
    claim_id: c.x.1
    claim_kind: definition
    claim_text: X is something.
    locator:
      locator_type: markdown_prose_v1
      heading_path: [Intro]
      paragraph_index: 1
      sentence_start: 1
      sentence_end: 1
"""


def test_parser_admits_code_v1_when_allowed() -> None:
    out = parse_claim_extraction_output(
        _CODE_OUTPUT, allowed_locator_types={"code_v1"}
    )
    assert [c.locator.locator_type for c in out] == ["code_v1"]


def test_parser_refuses_code_v1_when_only_narrative_allowed() -> None:
    with pytest.raises(ModelOutputError) as exc:
        parse_claim_extraction_output(
            _CODE_OUTPUT, allowed_locator_types={"markdown_prose_v1"}
        )
    msg = str(exc.value)
    assert "code_v1" in msg
    assert "markdown_prose_v1" in msg
    # And the legacy default-keyword path (no allowed set) still
    # refuses code_v1, matching the narrative-only contract.
    with pytest.raises(ModelOutputError):
        parse_claim_extraction_output(_CODE_OUTPUT)
    # Symmetric positive control: narrative output parses under the
    # narrative set and the legacy default.
    assert parse_claim_extraction_output(
        _NARRATIVE_OUTPUT, allowed_locator_types={"markdown_prose_v1"}
    )
    assert parse_claim_extraction_output(_NARRATIVE_OUTPUT)


def _seed_with_locator(**locator_fields) -> SeedClaim:
    return SeedClaim(
        entity_id="code.store",
        entity_type="concept",
        display_name="Store",
        claim_id="c.store.1",
        claim_kind="definition",
        claim_text="Store is a class.",
        locator=Locator(locator_type="code_v1", **locator_fields),
    )


def test_declaration_validator_accepts_match_and_refuses_fabricated_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub the structure extractor with a deterministic single
    declaration item; assert the validator admits an exact-match
    locator and refuses one that does not line up."""
    import importlib

    from llloom.structured.extract import StructureItem, StructureReport

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
                        "path": "raw/sources/store.py",
                        "start_line": 1,
                        "start_col": 1,
                        "end_line": 3,
                        "end_col": 20,
                    },
                )
            ],
        )

    monkeypatch.setattr(ingest_mod, "extract_structure", _fake_extract)

    good = _seed_with_locator(
        path="raw/sources/store.py",
        start_line=1,
        start_col=1,
        end_line=3,
        end_col=20,
    )
    _validate_code_v1_declaration_locators(
        candidates=[good],
        source_text=_PY_SOURCE,
        source_id="src.store",
        source_class="code",
        raw_path="raw/sources/store.py",
        content_hash="sha256:" + "a" * 64,
    )

    bad = _seed_with_locator(
        path="raw/sources/store.py",
        start_line=2,
        start_col=9,
        end_line=2,
        end_col=12,
    )
    with pytest.raises(CodeClaimContractError) as exc:
        _validate_code_v1_declaration_locators(
            candidates=[bad],
            source_text=_PY_SOURCE,
            source_id="src.store",
            source_class="code",
            raw_path="raw/sources/store.py",
            content_hash="sha256:" + "a" * 64,
        )
    assert "declaration" in str(exc.value).lower()


def test_provider_instructions_are_source_class_aware() -> None:
    """Both adapter prompts must describe the source-class-aware
    contract, not the old blanket 'never emit code_v1' rule. Verifies
    the narrative-forbidden phrasing and the declaration-only
    admission for code-backed ingest. A regression that reinstated
    the blanket rule would lose one of the anchor phrases."""
    for prompt in (_OPENAI_PROMPT, _ANTHROPIC_PROMPT):
        # Narrative-side rule still present (in some line break).
        narrative_clause = (
            "narrative sources" in prompt
            and "never emit\n  code_v1" in prompt
        )
        assert narrative_clause, prompt
        # Code-backed admission with the declaration-level qualifier
        # and the attached-explanation qualifier (this slice's
        # addition).
        assert "explicit code-backed claim_extract" in prompt
        assert "declaration-level spans" in prompt
        assert "attached explanation spans" in prompt
        # Detached comments and arbitrary body spans remain closed.
        assert "detached comments" in prompt
        assert "arbitrary code-body spans" in prompt
