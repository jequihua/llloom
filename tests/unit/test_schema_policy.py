"""Unit tests for schema loading and policy resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.schema.policy import INGEST_POLICIES, SchemaError, load_schema
from llloom.workspace.layout import Workspace


def test_starter_schema_loads(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    schema = load_schema(ws)
    assert "markdown_prose" in schema.source_classes
    assert schema.resolve_ingest_policy("markdown_prose") in INGEST_POLICIES
    assert schema.resolve_ingest_policy("markdown_prose") == "claim_extract_and_view_render"
    assert schema.resolve_ingest_policy("legal_act") == "claim_extract"
    assert schema.resolve_ingest_policy("code") == "structure_extract"
    assert schema.resolve_ingest_policy("structured_yaml") == "structure_extract"
    assert schema.unknown_policy == "deny"


def test_starter_schema_registers_raw_evidence_as_index_only(
    tmp_path: Path,
) -> None:
    """Slice 083: the neutral ``raw_evidence`` starter source class
    is registered in the starter schema, reuses the
    ``markdown_prose_v1`` locator shape (for schema compatibility),
    and resolves to the existing ``index_only`` ingest policy.

    The class is intentionally narrow — it is the "register and
    hash, retrieve exactly, no claims or structure or model"
    surface for UTF-8 text whose structure llloom does not yet
    parse. No new locator type is introduced; the slice's
    `markdown_prose_v1` reuse is documented in the schema
    description.
    """
    ws = Workspace.init(tmp_path)
    schema = load_schema(ws)
    assert "raw_evidence" in schema.source_classes
    raw_class = schema.source_classes["raw_evidence"]
    assert raw_class.locator == "markdown_prose_v1"
    assert schema.resolve_ingest_policy("raw_evidence") == "index_only"
    # Existing classes remain registered byte-identically.
    for existing in (
        "markdown_prose",
        "legal_act",
        "code",
        "structured_yaml",
    ):
        assert existing in schema.source_classes


def test_unknown_policy_rejected(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    (tmp_path / "schema" / "ingest_policies.yaml").write_text(
        "policies:\n  markdown_prose: wild_policy\n", encoding="utf-8"
    )
    with pytest.raises(SchemaError):
        load_schema(ws)


def test_policy_referencing_unknown_class_rejected(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    (tmp_path / "schema" / "ingest_policies.yaml").write_text(
        "policies:\n  ghost_class: claim_extract\n", encoding="utf-8"
    )
    with pytest.raises(SchemaError) as exc:
        load_schema(ws)
    assert "ghost_class" in str(exc.value)


def test_spine_is_recognized(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    schema = load_schema(ws)
    assert schema.is_spine("pages/overview.md")
    assert schema.is_spine("pages/navigation/anything.md")
    assert not schema.is_spine("pages/concepts/example.md")


def test_unknown_locator_rejected(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    (tmp_path / "schema" / "source_classes.yaml").write_text(
        "classes:\n  weird:\n    locator: not_a_real_locator\n",
        encoding="utf-8",
    )
    with pytest.raises(SchemaError) as exc:
        load_schema(ws)
    assert "unsupported locator" in str(exc.value)

