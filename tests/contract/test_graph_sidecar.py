"""Contract tests for the rebuildable SQLite graph sidecar.

The sidecar lives under ``state/graph/graph.sqlite`` and is derived
state only. It may accelerate neighbor lookup over canonical
``relations:`` records in entity YAML containers, but every returned
``GraphEdge`` must be rehydrated from canonical YAML before it
reaches a public result.
"""

from __future__ import annotations

import io
import json
import sqlite3
from contextlib import closing, redirect_stdout
from pathlib import Path

import pytest

from llloom.claims.models import (
    Assertion,
    EntityContainer,
    Evidence,
    Locator,
    Relation,
)
from llloom.claims.store import ClaimStore
from llloom.cli import main as cli_main
from llloom.llm.harness import LLMInvoke
from llloom.ops.rebuild import REBUILD_TARGETS, rebuild
from llloom.state.graph import (
    GraphEdge,
    GraphSidecarError,
    build_graph_sidecar,
    graph_neighbors,
    graph_sidecar_exists,
)
from llloom.workspace.layout import Workspace


def _make_assertion(claim_id: str, subject_id: str, text: str) -> Assertion:
    return Assertion(
        claim_id=claim_id,
        subject_id=subject_id,
        claim_kind="definition",
        claim_text=text,
        evidence=[
            Evidence(
                source_id="src.stub",
                locator=Locator(
                    locator_type="markdown_prose_v1",
                    heading_path=["Methods"],
                    paragraph_index=1,
                    sentence_start=1,
                    sentence_end=1,
                ),
                excerpt_hash="0" * 64,
            )
        ],
    )


def _seed_two_entity_workspace(tmp_path: Path) -> Workspace:
    """Seed two entities with one active relation between them.

    The relation is stored on the source entity (``concept.alpha``) and
    points at ``concept.beta``'s claim.
    """
    ws = Workspace.init(tmp_path)
    store = ClaimStore(ws)
    alpha = EntityContainer(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        assertions=[_make_assertion("c.alpha.1", "concept.alpha", "alpha claim")],
        relations=[
            Relation(
                relation_id="r.alpha_supports_beta",
                source_claim_id="c.alpha.1",
                relation_type="supports",
                target_claim_id="c.beta.1",
                status="active",
            )
        ],
    )
    beta = EntityContainer(
        entity_id="concept.beta",
        entity_type="concept",
        display_name="Beta",
        assertions=[_make_assertion("c.beta.1", "concept.beta", "beta claim")],
    )
    store.save_entity(alpha)
    store.save_entity(beta)
    return ws


# --- 1. rebuild creates sqlite DB with expected counts ------------------


def test_rebuild_graph_creates_sqlite_database_and_counts(tmp_path: Path) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    assert "graph" in REBUILD_TARGETS

    summary = rebuild(ws, target="graph")
    assert summary["target"] == "graph"
    assert summary["edge_rows"] == 1
    assert summary["index_path"] == ws.graph_db.as_posix()
    assert ws.graph_db.is_file()
    assert graph_sidecar_exists(ws)

    with closing(sqlite3.connect(str(ws.graph_db))) as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='edges'"
        ).fetchone()
        assert row is not None
        count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert count == 1


# --- 2. graph_neighbors returns a rehydrated canonical edge -------------


def test_graph_neighbors_returns_rehydrated_canonical_edge(tmp_path: Path) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    rebuild(ws, target="graph")

    edges = graph_neighbors(ws, claim_id="c.alpha.1", direction="out")
    assert len(edges) == 1
    edge = edges[0]
    assert isinstance(edge, GraphEdge)
    assert edge.relation_id == "r.alpha_supports_beta"
    assert edge.source_entity_id == "concept.alpha"
    assert edge.source_claim_id == "c.alpha.1"
    assert edge.target_entity_id == "concept.beta"
    assert edge.target_claim_id == "c.beta.1"
    assert edge.relation_type == "supports"
    assert edge.status == "active"


# --- 3. fallback without sidecar ----------------------------------------


def test_graph_neighbors_falls_back_without_sidecar(tmp_path: Path) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    assert not graph_sidecar_exists(ws)

    edges = graph_neighbors(ws, claim_id="c.alpha.1", direction="out")
    assert len(edges) == 1
    assert edges[0].relation_id == "r.alpha_supports_beta"
    assert edges[0].target_claim_id == "c.beta.1"


# --- 4. stale sidecar revalidates at lookup time ------------------------


def test_stale_sidecar_relation_not_emitted_after_canonical_change(
    tmp_path: Path,
) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    rebuild(ws, target="graph")
    assert graph_sidecar_exists(ws)

    # Flip the relation to an inactive status at the canonical level
    # but do NOT rebuild the sidecar. The sidecar row still points at
    # this relation; rehydration must drop it.
    store = ClaimStore(ws)
    alpha = store.load_entity("concept.alpha")
    alpha.relations[0].status = "retracted"
    store.save_entity(alpha)

    edges = graph_neighbors(ws, claim_id="c.alpha.1", direction="out")
    assert edges == []


def test_stale_sidecar_endpoint_not_emitted_after_claim_retraction(
    tmp_path: Path,
) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    rebuild(ws, target="graph")

    # Retract the target claim; the sidecar row is stale but
    # rehydration must catch it.
    store = ClaimStore(ws)
    beta = store.load_entity("concept.beta")
    beta.assertions[0].status = "retracted"
    store.save_entity(beta)

    edges = graph_neighbors(ws, claim_id="c.alpha.1", direction="out")
    assert edges == []


# --- 5. direction and relation-type filters -----------------------------


def test_graph_neighbor_direction_and_relation_type_filters(
    tmp_path: Path,
) -> None:
    ws = Workspace.init(tmp_path)
    store = ClaimStore(ws)
    alpha = EntityContainer(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        assertions=[_make_assertion("c.alpha.1", "concept.alpha", "alpha claim")],
        relations=[
            Relation(
                relation_id="r.a_supports_b",
                source_claim_id="c.alpha.1",
                relation_type="supports",
                target_claim_id="c.beta.1",
                status="active",
            ),
            Relation(
                relation_id="r.a_refines_g",
                source_claim_id="c.alpha.1",
                relation_type="refines",
                target_claim_id="c.gamma.1",
                status="active",
            ),
        ],
    )
    beta = EntityContainer(
        entity_id="concept.beta",
        entity_type="concept",
        display_name="Beta",
        assertions=[_make_assertion("c.beta.1", "concept.beta", "beta claim")],
        relations=[
            Relation(
                relation_id="r.b_about_a",
                source_claim_id="c.beta.1",
                relation_type="about",
                target_claim_id="c.alpha.1",
                status="active",
            )
        ],
    )
    gamma = EntityContainer(
        entity_id="concept.gamma",
        entity_type="concept",
        display_name="Gamma",
        assertions=[_make_assertion("c.gamma.1", "concept.gamma", "gamma claim")],
    )
    store.save_entity(alpha)
    store.save_entity(beta)
    store.save_entity(gamma)
    rebuild(ws, target="graph")

    out_edges = graph_neighbors(ws, claim_id="c.alpha.1", direction="out")
    assert {e.relation_id for e in out_edges} == {
        "r.a_supports_b",
        "r.a_refines_g",
    }

    in_edges = graph_neighbors(ws, claim_id="c.alpha.1", direction="in")
    assert {e.relation_id for e in in_edges} == {"r.b_about_a"}

    both_edges = graph_neighbors(ws, claim_id="c.alpha.1", direction="both")
    assert {e.relation_id for e in both_edges} == {
        "r.a_supports_b",
        "r.a_refines_g",
        "r.b_about_a",
    }

    supports_only = graph_neighbors(
        ws,
        claim_id="c.alpha.1",
        direction="both",
        relation_types={"supports"},
    )
    assert {e.relation_id for e in supports_only} == {"r.a_supports_b"}

    with pytest.raises(GraphSidecarError):
        graph_neighbors(ws, claim_id="c.alpha.1", direction="sideways")


# --- 6. skip relations whose target claim is missing --------------------


def test_graph_rebuild_skips_missing_endpoint_claims(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    store = ClaimStore(ws)
    alpha = EntityContainer(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        assertions=[_make_assertion("c.alpha.1", "concept.alpha", "alpha claim")],
        relations=[
            Relation(
                relation_id="r.dangling",
                source_claim_id="c.alpha.1",
                relation_type="supports",
                target_claim_id="c.nonexistent.1",
                status="active",
            )
        ],
    )
    store.save_entity(alpha)

    summary = rebuild(ws, target="graph")
    assert summary["edge_rows"] == 0

    with closing(sqlite3.connect(str(ws.graph_db))) as conn:
        count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert count == 0


# --- 7. LLM harness is untouched by graph operations --------------------


def test_query_and_llm_harness_are_untouched_by_graph_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _seed_two_entity_workspace(tmp_path)

    def _fail_invoke(self, **kwargs):  # noqa: ANN001 - test stub
        raise AssertionError(
            "graph sidecar must not invoke LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r}"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _fail_invoke)

    rebuild(ws, target="graph")
    edges = graph_neighbors(ws, claim_id="c.alpha.1", direction="out")
    assert len(edges) == 1


# --- 8. CLI smoke: `llloom rebuild graph` -------------------------------


def test_cli_rebuild_graph(tmp_path: Path) -> None:
    ws = _seed_two_entity_workspace(tmp_path)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(["--root", str(ws.root), "rebuild", "graph"])
    assert rc == 0

    payload = json.loads(buf.getvalue())
    assert payload["target"] == "graph"
    assert payload["edge_rows"] == 1
    assert ws.graph_db.is_file()


# --- 9. graph_neighbors refuses non-positive limit -----------------------


def test_graph_neighbors_non_positive_limit_returns_empty(tmp_path: Path) -> None:
    """``limit <= 0`` returns ``[]`` on both the fallback (no sidecar) and
    the sidecar-present paths. Positive ``limit`` still returns edges."""
    ws = _seed_two_entity_workspace(tmp_path)

    assert not graph_sidecar_exists(ws)
    assert graph_neighbors(ws, claim_id="c.alpha.1", direction="out", limit=0) == []
    assert graph_neighbors(ws, claim_id="c.alpha.1", direction="out", limit=-1) == []

    rebuild(ws, target="graph")
    assert graph_sidecar_exists(ws)
    assert graph_neighbors(ws, claim_id="c.alpha.1", direction="out", limit=0) == []
    assert graph_neighbors(ws, claim_id="c.alpha.1", direction="out", limit=-1) == []

    # Positive-control: limit=1 still returns the one canonical edge.
    positive = graph_neighbors(ws, claim_id="c.alpha.1", direction="out", limit=1)
    assert len(positive) == 1
    assert positive[0].relation_id == "r.alpha_supports_beta"


# --- 10. rebuild determinism: identical counts, no duplicate rows --------


def test_rebuild_graph_is_deterministic(tmp_path: Path) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    a = rebuild(ws, target="graph")
    b = rebuild(ws, target="graph")
    assert a["edge_rows"] == b["edge_rows"] == 1

    with closing(sqlite3.connect(str(ws.graph_db))) as conn:
        rows = conn.execute(
            "SELECT relation_id, source_entity_id, source_claim_id, "
            "relation_type, target_entity_id, target_claim_id, status "
            "FROM edges ORDER BY relation_id"
        ).fetchall()
    assert rows == [
        (
            "r.alpha_supports_beta",
            "concept.alpha",
            "c.alpha.1",
            "supports",
            "concept.beta",
            "c.beta.1",
            "active",
        )
    ]
