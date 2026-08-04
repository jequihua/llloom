"""Contract tests for the structure-report graph sidecar.

Extends the existing graph sidecar at ``state/graph/graph.sqlite``
with a second derived table, ``structure_edges``, indexing direct
parent/child containment edges from the on-disk structure reports
under ``state/structure/<source_id>.yaml``.

Every emitted ``StructureGraphEdge`` is rehydrated from the current
report and revalidated against the current source registry record;
the SQLite row is used only to narrow candidate pairs. Stale rows
drop silently.

The pre-existing claim-relation graph behavior (``edges`` table,
``GraphEdge``, ``graph_neighbors``) must remain unchanged.
"""

from __future__ import annotations

import io
import json
import sqlite3
from contextlib import closing, redirect_stdout
from pathlib import Path

import pytest

from llloom.cli import main as cli_main
from llloom.llm.harness import LLMInvoke
from llloom.ops.ingest import ingest
from llloom.ops.rebuild import rebuild
from llloom.sources.registry import SourceRegistry
from llloom.state.graph import (
    GraphSidecarError,
    StructureGraphEdge,
    structure_graph_neighbors,
)
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


# --- 1. rebuild graph indexes structure edges and reports the count -----


def test_rebuild_graph_indexes_structure_report_edges(tmp_path: Path) -> None:
    ws, _ = _seed_yaml_source(tmp_path)
    summary = rebuild(ws, target="graph")

    assert summary["target"] == "graph"
    assert "edge_rows" in summary
    assert summary["structure_edge_rows"] == 3

    with closing(sqlite3.connect(str(ws.graph_db))) as conn:
        rows = conn.execute(
            "SELECT source_id, parent_symbol_path, child_symbol_path, "
            "child_kind, child_name, report_path "
            "FROM structure_edges ORDER BY parent_symbol_path, child_symbol_path"
        ).fetchall()
    expected_report = "state/structure/src.policies.yaml"
    assert rows == [
        (
            "src.policies",
            "defaults",
            "defaults.unknown",
            "mapping_key",
            "unknown",
            expected_report,
        ),
        (
            "src.policies",
            "policies",
            "policies.legal_act",
            "mapping_key",
            "legal_act",
            expected_report,
        ),
        (
            "src.policies",
            "policies",
            "policies.markdown_prose",
            "mapping_key",
            "markdown_prose",
            expected_report,
        ),
    ]


# --- 2. direction=out returns child edges of a parent symbol ------------


def test_structure_graph_neighbors_direction_out_returns_children(
    tmp_path: Path,
) -> None:
    ws, _ = _seed_yaml_source(tmp_path)
    rebuild(ws, target="graph")

    edges = structure_graph_neighbors(
        ws,
        source_id="src.policies",
        symbol_path="policies",
        direction="out",
    )
    assert all(isinstance(e, StructureGraphEdge) for e in edges)
    assert [e.child_symbol_path for e in edges] == [
        "policies.markdown_prose",
        "policies.legal_act",
    ]
    for edge in edges:
        assert edge.source_id == "src.policies"
        assert edge.source_class == "structured_yaml"
        assert edge.language == "yaml"
        assert edge.parent_symbol_path == "policies"
        assert edge.child_kind == "mapping_key"
        assert edge.report_path == "state/structure/src.policies.yaml"


# --- 3. direction=in returns the parent edge for a child symbol ---------


def test_structure_graph_neighbors_direction_in_returns_parent(
    tmp_path: Path,
) -> None:
    ws, _ = _seed_yaml_source(tmp_path)
    rebuild(ws, target="graph")

    edges = structure_graph_neighbors(
        ws,
        source_id="src.policies",
        symbol_path="policies.markdown_prose",
        direction="in",
    )
    assert len(edges) == 1
    edge = edges[0]
    assert edge.parent_symbol_path == "policies"
    assert edge.child_symbol_path == "policies.markdown_prose"
    assert edge.child_name == "markdown_prose"


# --- 4. direction=both returns the union -------------------------------


def test_structure_graph_neighbors_direction_both_returns_union(
    tmp_path: Path,
) -> None:
    ws, _ = _seed_yaml_source(tmp_path)
    rebuild(ws, target="graph")

    edges = structure_graph_neighbors(
        ws,
        source_id="src.policies",
        symbol_path="policies",
        direction="both",
    )
    # "policies" is a root, so only "out" edges (to its two children)
    # exist; "in" contributes nothing here. Document this directly.
    assert {(e.parent_symbol_path, e.child_symbol_path) for e in edges} == {
        ("policies", "policies.markdown_prose"),
        ("policies", "policies.legal_act"),
    }


# --- 5. fallback without a sidecar still returns correct edges ---------


def test_structure_graph_neighbors_falls_back_without_sidecar(
    tmp_path: Path,
) -> None:
    ws, _ = _seed_yaml_source(tmp_path)
    # Do NOT call rebuild; the sidecar must be absent.
    assert not ws.graph_db.is_file()

    edges = structure_graph_neighbors(
        ws,
        source_id="src.policies",
        symbol_path="policies",
        direction="out",
    )
    assert {(e.parent_symbol_path, e.child_symbol_path) for e in edges} == {
        ("policies", "policies.markdown_prose"),
        ("policies", "policies.legal_act"),
    }


# --- 6. stale sidecar drops when the report is deleted ----------------


def test_stale_structure_edges_drop_when_report_is_deleted(
    tmp_path: Path,
) -> None:
    ws, _ = _seed_yaml_source(tmp_path)
    rebuild(ws, target="graph")
    assert ws.graph_db.is_file()

    # Sidecar rows still reference the report. Delete the report
    # without rebuilding the sidecar.
    ws.structure_report_path("src.policies").unlink()

    edges = structure_graph_neighbors(
        ws,
        source_id="src.policies",
        symbol_path="policies",
        direction="both",
    )
    assert edges == []


# --- 7. stale sidecar drops when the source is retracted --------------


def test_stale_structure_edges_drop_when_source_is_retracted(
    tmp_path: Path,
) -> None:
    ws, _ = _seed_yaml_source(tmp_path)
    rebuild(ws, target="graph")

    registry = SourceRegistry(ws)
    registry.mark_retracted("src.policies", reason="contract-test")

    edges = structure_graph_neighbors(
        ws,
        source_id="src.policies",
        symbol_path="policies",
        direction="both",
    )
    assert edges == []


# --- 8. metadata only — no scalar values or comments leak --------------


def test_structure_edge_rows_contain_metadata_only(tmp_path: Path) -> None:
    """A poison scalar value and a poison comment planted in the YAML
    source must not appear in any column of any ``structure_edges``
    row. Structure edges carry parent/child symbol paths, kind/name
    metadata, and the report path — never source bodies, values, or
    comments."""
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "poisoned.yaml"
    src.write_text(
        "# POISON_COMMENT_abc\n"
        "policies:\n"
        "  markdown_prose: POISON_VALUE_xyz\n",
        encoding="utf-8",
    )
    ingest(ws, src, source_id="src.poisoned", source_class="structured_yaml")
    rebuild(ws, target="graph")

    with closing(sqlite3.connect(str(ws.graph_db))) as conn:
        rows = conn.execute("SELECT * FROM structure_edges").fetchall()
    joined = "\n".join(str(cell) for row in rows for cell in row)
    assert "policies" in joined  # positive control: symbol paths indexed
    assert "POISON_VALUE_xyz" not in joined
    assert "POISON_COMMENT_abc" not in joined

    # And no rehydrated edge surfaces poison either.
    edges = structure_graph_neighbors(
        ws,
        source_id="src.poisoned",
        symbol_path="policies",
        direction="out",
    )
    for edge in edges:
        for value in (
            edge.parent_symbol_path,
            edge.child_symbol_path,
            edge.child_kind,
            edge.child_name,
            edge.source_id,
            edge.source_class,
            edge.language,
            edge.report_path,
        ):
            assert "POISON_VALUE_xyz" not in value
            assert "POISON_COMMENT_abc" not in value


# --- 9. unknown direction raises; limit<=0 returns [] ------------------


def test_structure_graph_neighbors_input_validation(tmp_path: Path) -> None:
    ws, _ = _seed_yaml_source(tmp_path)
    rebuild(ws, target="graph")

    with pytest.raises(GraphSidecarError):
        structure_graph_neighbors(
            ws,
            source_id="src.policies",
            symbol_path="policies",
            direction="sideways",
        )

    assert (
        structure_graph_neighbors(
            ws,
            source_id="src.policies",
            symbol_path="policies",
            direction="out",
            limit=0,
        )
        == []
    )
    assert (
        structure_graph_neighbors(
            ws,
            source_id="src.policies",
            symbol_path="policies",
            direction="out",
            limit=-1,
        )
        == []
    )

    # Positive control: limit=1 returns exactly one canonical edge.
    one = structure_graph_neighbors(
        ws,
        source_id="src.policies",
        symbol_path="policies",
        direction="out",
        limit=1,
    )
    assert len(one) == 1


# --- 10. no LLM invoke from this path; CLI smoke includes new field ---


def test_structure_graph_path_does_not_invoke_llm_and_cli_emits_new_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, _ = _seed_yaml_source(tmp_path)

    def _fail_invoke(self, **kwargs):  # noqa: ANN001 - test stub
        raise AssertionError(
            "structure-graph sidecar must not invoke LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r}"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _fail_invoke)

    rebuild(ws, target="graph")
    edges = structure_graph_neighbors(
        ws,
        source_id="src.policies",
        symbol_path="policies",
        direction="out",
    )
    assert len(edges) == 2  # two children of "policies"

    # CLI smoke: the additive `structure_edge_rows` key appears.
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(["--root", str(ws.root), "rebuild", "graph"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["target"] == "graph"
    assert payload["structure_edge_rows"] == 3
    assert "edge_rows" in payload  # claim-relation behavior preserved
