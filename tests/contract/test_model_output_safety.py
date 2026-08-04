"""Contract tests for the strict model-output contract.

Failure modes that must be safe (no partial persistence, visible
refusal in the result and on the operation journal):

- malformed YAML
- top-level type that is not a mapping
- missing top-level ``claims`` key
- ``claims`` is not a list
- per-candidate missing required field
- per-candidate unknown field
- per-candidate locator missing or malformed
- locator that does not resolve against the current source
- explicit ``excerpt_hash`` that mismatches the computed hash

Plus regression coverage that the prior safety contracts still hold:

- ingest still refuses ``ClaimBlockRegion`` inputs at the harness boundary
- the model-backed extraction path persists a single invocation log
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.claims.store import ClaimStore
from llloom.llm.harness import (
    ClaimBlockRegion,
    HarnessRefusal,
    LLMInvoke,
    SourceDocument,
)
from llloom.llm.output import (
    ModelOutputError,
    parse_claim_extraction_output,
)
from llloom.ops.ingest import ingest
from llloom.state.journal import OperationJournal
from llloom.workspace.layout import Workspace


SOURCE = """\
# Article

## Methods

A canonical sentence in paragraph one. Another sentence here.
"""


class _FakeModel:
    identifier = "fake-test-model/v0"

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, prompt: str) -> str:
        _ = prompt
        return self._output


def _ws_and_source(tmp_path: Path) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE, encoding="utf-8")
    return ws, src


def _ingest_with_output(tmp_path: Path, output: str):
    ws, src = _ws_and_source(tmp_path)
    result = ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        llm=LLMInvoke(model=_FakeModel(output)),
    )
    return ws, result


# ---- direct parser-level checks -----------------------------------------


def test_parser_empty_output_is_zero_candidates() -> None:
    assert parse_claim_extraction_output("") == []
    assert parse_claim_extraction_output("   \n  \n") == []


def test_parser_rejects_invalid_yaml() -> None:
    with pytest.raises(ModelOutputError) as exc:
        parse_claim_extraction_output(":\n  not: [valid")
    assert "YAML parse error" in str(exc.value)


def test_parser_rejects_top_level_list() -> None:
    with pytest.raises(ModelOutputError) as exc:
        parse_claim_extraction_output("- foo\n- bar\n")
    assert "top-level" in str(exc.value)


def test_parser_rejects_missing_claims_key() -> None:
    with pytest.raises(ModelOutputError) as exc:
        parse_claim_extraction_output("other_key: 1\n")
    assert "claims" in str(exc.value)


def test_parser_rejects_claims_not_a_list() -> None:
    with pytest.raises(ModelOutputError):
        parse_claim_extraction_output("claims:\n  not_a_list: true\n")


def test_parser_rejects_unknown_field() -> None:
    text = """\
claims:
  - entity_id: e1
    entity_type: concept
    display_name: X
    claim_id: c1
    claim_kind: definition
    claim_text: text
    locator:
      locator_type: markdown_prose_v1
    extra_field: not_allowed
"""
    with pytest.raises(ModelOutputError) as exc:
        parse_claim_extraction_output(text)
    assert "unknown" in str(exc.value)


def test_parser_rejects_missing_required_field() -> None:
    text = """\
claims:
  - entity_id: e1
    entity_type: concept
    # display_name omitted
    claim_id: c1
    claim_kind: definition
    claim_text: text
    locator:
      locator_type: markdown_prose_v1
"""
    with pytest.raises(ModelOutputError) as exc:
        parse_claim_extraction_output(text)
    assert "display_name" in str(exc.value)


def test_parser_rejects_missing_locator() -> None:
    text = """\
claims:
  - entity_id: e1
    entity_type: concept
    display_name: X
    claim_id: c1
    claim_kind: definition
    claim_text: text
"""
    with pytest.raises(ModelOutputError) as exc:
        parse_claim_extraction_output(text)
    assert "locator" in str(exc.value)


def test_locator_is_not_in_optional_field_group() -> None:
    """Shape-level invariant from the post-slice cleanup pass.

    ``locator`` is mandatory per the spec and per parser behavior; it
    must not appear in ``_OPTIONAL_FIELDS`` so a future maintainer
    cannot misread the field groups. ``locator`` lives in its own
    ``_REQUIRED_MAPPING_FIELDS`` group, separate from the scalar
    required fields that the per-field string check loops over.
    """
    from llloom.llm import output as out_mod

    assert "locator" not in out_mod._OPTIONAL_FIELDS
    assert "locator" in out_mod._REQUIRED_MAPPING_FIELDS
    assert "locator" in out_mod._KNOWN_FIELDS


# ---- ingest-level: malformed output is a batch-atomic refusal ----------


def test_ingest_with_malformed_yaml_persists_nothing(tmp_path: Path) -> None:
    ws, result = _ingest_with_output(tmp_path, ":\n  not: [valid")
    assert not result.succeeded
    assert "extraction failed" in result.refusal_reason
    assert result.extraction_errors, (
        "extraction_errors should record what went wrong"
    )
    assert ClaimStore(ws).list_entity_ids() == []
    # The operation journal should record the parse error in notes.
    journal = OperationJournal(ws)
    entry = journal.load(result.op_id)
    assert any("parse" in note for note in entry.notes)


def test_ingest_with_unresolvable_locator_persists_nothing(tmp_path: Path) -> None:
    bad = """\
claims:
  - entity_id: concept.x
    entity_type: concept
    display_name: X
    claim_id: c.x.1
    claim_kind: definition
    claim_text: not present in source
    locator:
      locator_type: markdown_prose_v1
      heading_path: ["NoSuchHeading"]
      paragraph_index: 99
      sentence_start: 1
      sentence_end: 1
"""
    ws, result = _ingest_with_output(tmp_path, bad)
    assert not result.succeeded
    assert "extraction failed" in result.refusal_reason
    assert any("c.x.1" in note for note in result.extraction_errors)
    assert ClaimStore(ws).list_entity_ids() == []


def test_ingest_with_mismatched_excerpt_hash_persists_nothing(tmp_path: Path) -> None:
    bad = """\
claims:
  - entity_id: concept.x
    entity_type: concept
    display_name: X
    claim_id: c.x.1
    claim_kind: definition
    claim_text: A canonical sentence in paragraph one.
    excerpt_hash: "sha256:deadbeefdeadbeef"
    locator:
      locator_type: markdown_prose_v1
      heading_path: ["Methods"]
      paragraph_index: 1
      sentence_start: 1
      sentence_end: 1
"""
    ws, result = _ingest_with_output(tmp_path, bad)
    assert not result.succeeded
    assert "extraction failed" in result.refusal_reason
    assert any("excerpt_hash" in note for note in result.extraction_errors)
    assert ClaimStore(ws).list_entity_ids() == []


# ---- regression coverage on prior safety contracts ---------------------


def test_ingest_still_refuses_claim_block_inputs_to_harness() -> None:
    """Independent of model output, the harness must refuse
    ``ClaimBlockRegion`` on ingest. This guards against any future
    regression that wires page regions back into ingest."""
    harness = LLMInvoke()
    block = ClaimBlockRegion(
        page_id="concept/x", block_id="block_x", rendered_text="prior page text"
    )
    with pytest.raises(HarnessRefusal):
        harness.invoke(
            op_id="op.ingest.guard",
            operation_kind="ingest",
            claim_blocks=[block],
        )


def test_model_backed_extraction_persists_one_invocation_log(tmp_path: Path) -> None:
    """Even with non-empty model output, the journal entry persists
    exactly one invocation log per ingest, with summaries only."""
    good = """\
claims:
  - entity_id: concept.x
    entity_type: concept
    display_name: X
    claim_id: c.x.1
    claim_kind: definition
    claim_text: A canonical sentence in paragraph one.
    locator:
      locator_type: markdown_prose_v1
      heading_path: ["Methods"]
      paragraph_index: 1
      sentence_start: 1
      sentence_end: 1
"""
    ws, result = _ingest_with_output(tmp_path, good)
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)

    journal = OperationJournal(ws)
    entry = journal.load(result.op_id)
    assert len(entry.invocation_logs) == 1
    log = entry.invocation_logs[0]
    assert log["operation_kind"] == "ingest"
    classes = sorted(item["class"] for item in log["read_inputs"])
    assert "SourceDocument" in classes
    assert "SchemaDocument" in classes

    # Persisted journal must not contain raw source text.
    journal_yaml = (ws.state_journals / f"{result.op_id}.yaml").read_text(
        encoding="utf-8"
    )
    assert "A canonical sentence in paragraph one." not in journal_yaml, (
        "raw source text leaked into persisted invocation log; the audit "
        "trail must contain content hashes only"
    )
