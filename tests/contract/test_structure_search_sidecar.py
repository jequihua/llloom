"""Contract tests for structure-report search-sidecar integration.

The search sidecar indexes structure-report items as derived
``structure_item`` rows containing metadata only. `query`
rehydrates those rows from the on-disk report under
``state/structure/<source_id>.yaml`` before emitting any
``StructureItemHit``; a stale SQLite row never reaches
``QueryResult``.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
import yaml as yaml_mod

from llloom.llm.harness import LLMInvoke
from llloom.ops.ingest import ingest
from llloom.ops.query import query
from llloom.ops.rebuild import rebuild
from llloom.sources.registry import SourceRegistry
from llloom.workspace.layout import Workspace


YAML_SOURCE = (
    "policies:\n"
    "  markdown_prose: claim_extract_and_view_render\n"
    "  legal_act: claim_extract\n"
    "defaults:\n"
    "  unknown: deny\n"
)


def _seed_yaml_source(tmp_path: Path) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "policies.yaml"
    src.write_text(YAML_SOURCE, encoding="utf-8")
    ingest(ws, src, source_id="src.policies", source_class="structured_yaml")
    return ws, src


# --- 1. rebuild search indexes structure-report items -------------------


def test_rebuild_search_indexes_structure_report_items(tmp_path: Path) -> None:
    ws, _src = _seed_yaml_source(tmp_path)
    summary = rebuild(ws, target="search")
    assert summary["target"] == "search"
    assert "structure_rows" in summary
    expected_items = len(
        yaml_mod.safe_load(
            ws.structure_report_path("src.policies").read_text(encoding="utf-8")
        )["items"]
    )
    assert summary["structure_rows"] == expected_items

    with closing(sqlite3.connect(str(ws.search_db))) as conn:
        rows = conn.execute(
            "SELECT doc_type, source_id, structure_symbol_path "
            "FROM docs WHERE doc_type = 'structure_item'"
        ).fetchall()
    assert rows, "expected at least one structure_item row"
    assert all(r[0] == "structure_item" for r in rows)
    assert all(r[1] == "src.policies" for r in rows)
    assert {r[2] for r in rows} >= {"policies", "policies.markdown_prose"}


# --- 2. indexed text carries metadata only ------------------------------


def test_structure_sidecar_text_contains_only_structure_metadata(
    tmp_path: Path,
) -> None:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "poisoned.yaml"
    src.write_text(
        "# POISON_COMMENT_abc\n"
        "policies:\n"
        "  markdown_prose: POISON_VALUE_xyz\n",
        encoding="utf-8",
    )
    ingest(ws, src, source_id="src.poisoned", source_class="structured_yaml")
    rebuild(ws, target="search")

    with closing(sqlite3.connect(str(ws.search_db))) as conn:
        rows = conn.execute(
            "SELECT text FROM docs WHERE doc_type = 'structure_item'"
        ).fetchall()
    assert rows, "expected structure_item rows"
    joined = "\n".join(r[0] for r in rows)
    # Symbol paths are indexed...
    assert "markdown_prose" in joined
    assert "policies" in joined
    # ...but scalar values and comments must not leak.
    assert "POISON_VALUE_xyz" not in joined
    assert "POISON_COMMENT_abc" not in joined


# --- 3. query with sidecar returns rehydrated structure items ----------


def test_query_with_sidecar_returns_structure_items(tmp_path: Path) -> None:
    ws, _src = _seed_yaml_source(tmp_path)
    rebuild(ws, target="search")
    result = query(ws, question="markdown_prose")

    assert result.used_structure_items, "expected at least one structure item hit"
    found = next(
        (
            it
            for it in result.used_structure_items
            if it.symbol_path == "policies.markdown_prose"
        ),
        None,
    )
    assert found is not None
    assert found.source_id == "src.policies"
    assert found.source_class == "structured_yaml"
    assert found.language == "yaml"
    assert found.kind == "mapping_key"
    assert found.name == "markdown_prose"
    assert found.report_path == "state/structure/src.policies.yaml"
    assert found.locator["locator_type"] == "code_v1"
    # Deterministic answer line references the structure item.
    assert "structure item" in result.answer
    assert "policies.markdown_prose" in result.answer


# --- 4. rehydration uses the report, not the SQLite row -----------------


def test_query_structure_items_are_rehydrated_from_report_not_sidecar(
    tmp_path: Path,
) -> None:
    ws, _src = _seed_yaml_source(tmp_path)
    rebuild(ws, target="search")

    # Mutate the sidecar row metadata with a poison value.
    poison = "POISON_NAME_zzz"
    with closing(sqlite3.connect(str(ws.search_db))) as conn:
        conn.execute(
            "UPDATE docs SET structure_name = ? "
            "WHERE doc_type = 'structure_item' "
            "AND structure_symbol_path = 'policies.markdown_prose'",
            (poison,),
        )
        conn.commit()

    result = query(ws, question="markdown_prose")
    # The hit whose sidecar row was poisoned should be filtered out by
    # the structure_name guard; the canonical report value never
    # matches the poisoned sidecar value.
    for hit in result.used_structure_items:
        assert hit.symbol_path != "policies.markdown_prose"
        assert hit.name != poison
    # The poison value must never appear in the emitted answer.
    assert poison not in result.answer


# --- 5. stale sidecar rows drop when report/source disappears -----------


def test_stale_structure_sidecar_row_dropped_when_report_deleted(
    tmp_path: Path,
) -> None:
    ws, _src = _seed_yaml_source(tmp_path)
    rebuild(ws, target="search")

    # Sanity: before we mess with anything, the hit appears.
    before = query(ws, question="markdown_prose")
    assert any(
        it.symbol_path == "policies.markdown_prose"
        for it in before.used_structure_items
    )

    # Delete the structure report without rebuilding the sidecar.
    ws.structure_report_path("src.policies").unlink()
    result = query(ws, question="markdown_prose")
    assert result.used_structure_items == []


def test_stale_structure_sidecar_row_dropped_when_source_retracted(
    tmp_path: Path,
) -> None:
    ws, _src = _seed_yaml_source(tmp_path)
    rebuild(ws, target="search")
    SourceRegistry(ws).mark_retracted("src.policies", reason="test")
    result = query(ws, question="markdown_prose")
    assert result.used_structure_items == []


# --- 6. query still never invokes LLMInvoke -----------------------------


def test_query_structure_sidecar_does_not_invoke_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, _src = _seed_yaml_source(tmp_path)
    rebuild(ws, target="search")

    def _fail(self, **kwargs):  # noqa: ANN001 - test stub
        raise AssertionError(
            "query with structure-search sidecar must not invoke LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r}"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _fail)
    result = query(ws, question="markdown_prose")
    assert result.used_structure_items, (
        "expected at least one structure item to prove the tool ran end-to-end"
    )


# --- 7. query without sidecar has empty structure items -----------------


def test_query_without_sidecar_has_empty_structure_items(tmp_path: Path) -> None:
    ws, _src = _seed_yaml_source(tmp_path)
    # Do NOT rebuild search.
    assert not ws.search_db.exists()
    result = query(ws, question="markdown_prose")
    assert result.used_structure_items == []
    # Prior query behavior preserved.
    assert result.question == "markdown_prose"
    assert isinstance(result.citations, list)
