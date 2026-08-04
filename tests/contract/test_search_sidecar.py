"""Contract tests for the rebuildable SQLite FTS5 search sidecar.

The sidecar lives under ``state/search/search.sqlite`` and is derived
state only. It may accelerate candidate selection for ``query``, but
every emitted citation / verbatim span must still be rehydrated from
canonical YAML claim containers or raw registered source files.

These tests enforce the structural and safety properties directly;
integration-level query behavior lives in
``tests/integration/test_search_sidecar_query.py``.
"""

from __future__ import annotations

import io
import json
import sqlite3
from contextlib import closing, redirect_stdout
from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.cli import main as cli_main
from llloom.llm.harness import LLMInvoke
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.lint import FIXED_CANARY_TOKEN
from llloom.ops.query import query
from llloom.ops.rebuild import REBUILD_TARGETS, rebuild
from llloom.sources.registry import SourceRegistry
from llloom.state.search import (
    SearchSidecarError,
    build_search_sidecar,
    search_candidates,
    sidecar_exists,
)
from llloom.workspace.layout import Workspace


SOURCE = """\
# Article

## Methods

Complementarity prioritizes sites that add features not already represented in the selected set.
"""

PAGE_TEMPLATE = """\
---
page_id: concept/complementarity
page_class: concept
write_policy: mixed
status: rendered
---

<!-- llloom:claim-block id=claim_block.concept.complementarity -->
## Complementarity

Original rendered content.
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.complementarity owner=human -->
Human commentary body.
<!-- /llloom:commentary -->
"""


def _seed_workspace_with_claim(tmp_path: Path) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE, encoding="utf-8")
    page_path = ws.pages / "concepts" / "complementarity.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    seed = SeedClaim(
        entity_id="concept.complementarity",
        entity_type="concept",
        display_name="Complementarity",
        claim_id="c.cmp.1",
        claim_kind="definition",
        claim_text=(
            "Complementarity prioritizes sites that add features not already "
            "represented in the selected set."
        ),
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=(
            "concept/complementarity",
            "claim_block.concept.complementarity",
        ),
    )
    ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[seed],
    )
    return ws, src


def _wire_index_only(ws: Workspace) -> None:
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  sensitive:\n"
        "    locator: markdown_prose_v1\n",
        encoding="utf-8",
    )
    (ws.schema / "ingest_policies.yaml").write_text(
        "policies:\n"
        "  markdown_prose: claim_extract_and_view_render\n"
        "  sensitive: index_only\n"
        "defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )


# --- 1. rebuild creates FTS5 table and returns expected counts ----------


def test_rebuild_search_creates_fts5_database_and_counts(tmp_path: Path) -> None:
    ws, _src = _seed_workspace_with_claim(tmp_path)
    assert "search" in REBUILD_TARGETS

    summary = rebuild(ws, target="search")
    assert summary["target"] == "search"
    assert summary["claim_rows"] == 1
    assert summary["source_rows"] == 0
    assert ws.search_db.is_file()
    assert sidecar_exists(ws)

    # The docs table is an FTS5 virtual table with the expected columns.
    with closing(sqlite3.connect(str(ws.search_db))) as conn:
        cur = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='docs'"
        )
        row = cur.fetchone()
        assert row is not None
        assert "fts5" in row[0].lower()
        count = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        assert count == 1


# --- 2. retracted/archived assertions are skipped -----------------------


def test_retracted_and_archived_claims_are_not_indexed(tmp_path: Path) -> None:
    ws, _src = _seed_workspace_with_claim(tmp_path)
    # Mutate the entity YAML to flip the claim to retracted, bypassing
    # the cascade path (we're testing the index filter directly).
    from llloom.claims.store import ClaimStore

    store = ClaimStore(ws)
    entity = store.load_entity("concept.complementarity")
    entity.assertions[0].status = "retracted"
    store.save_entity(entity)

    summary = rebuild(ws, target="search")
    assert summary["claim_rows"] == 0

    # Archived also skipped.
    entity.assertions[0].status = "archived"
    store.save_entity(entity)
    summary = rebuild(ws, target="search")
    assert summary["claim_rows"] == 0


# --- 3. registered non-retracted index_only sources indexed; retracted skipped --


def test_index_only_sources_indexed_unless_retracted(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    _wire_index_only(ws)
    src = ws.raw_sources / "contract.md"
    src.write_text(
        "# Contract\n\nNet-30 with a 2% early-payment discount.\n",
        encoding="utf-8",
    )
    ingest(ws, src, source_id="src.contract", source_class="sensitive")

    summary = rebuild(ws, target="search")
    assert summary["source_rows"] == 1
    assert summary["claim_rows"] == 0

    SourceRegistry(ws).mark_retracted("src.contract", reason="test")
    summary = rebuild(ws, target="search")
    assert summary["source_rows"] == 0


# --- 4. stale sidecar rows are revalidated at query time ----------------


def test_stale_sidecar_row_not_emitted_after_claim_retraction(tmp_path: Path) -> None:
    """Build the index with a claim present, then retract the claim.
    The sidecar row still points at the retracted claim id, but query
    must rehydrate and drop it."""
    ws, _src = _seed_workspace_with_claim(tmp_path)
    rebuild(ws, target="search")
    assert sidecar_exists(ws)

    # Retract the claim at the canonical level. The sidecar row is
    # intentionally NOT rebuilt — this proves rehydration + status
    # filtering catches staleness.
    from llloom.claims.store import ClaimStore

    store = ClaimStore(ws)
    entity = store.load_entity("concept.complementarity")
    entity.assertions[0].status = "retracted_by_source"
    store.save_entity(entity)

    result = query(ws, question="complementarity sites features")
    assert result.used_claim_ids == []
    assert result.citations == []


def test_stale_sidecar_row_not_emitted_after_source_retraction(
    tmp_path: Path,
) -> None:
    ws = Workspace.init(tmp_path)
    _wire_index_only(ws)
    src = ws.raw_sources / "contract.md"
    src.write_text(
        "# Contract\n\nNet-30 with a 2% early-payment discount.\n",
        encoding="utf-8",
    )
    ingest(ws, src, source_id="src.contract", source_class="sensitive")
    rebuild(ws, target="search")

    SourceRegistry(ws).mark_retracted("src.contract", reason="test")

    result = query(ws, question="early-payment discount")
    # Retracted index_only source must not contribute spans even
    # though the sidecar still has a row for it.
    assert result.used_verbatim_spans == []


# --- 5. query with sidecar never invokes LLMInvoke ----------------------


def test_query_with_sidecar_does_not_invoke_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, _src = _seed_workspace_with_claim(tmp_path)
    rebuild(ws, target="search")
    assert sidecar_exists(ws)

    def _fail_invoke(self, **kwargs):  # noqa: ANN001 - test stub
        raise AssertionError(
            "query with search sidecar must not call LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r}"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _fail_invoke)

    result = query(ws, question="complementarity sites features")
    assert result.used_claim_ids, (
        "expected the sidecar-backed query to rehydrate at least one claim"
    )


# --- 6. commentary/spine tokens never appear in sidecar-backed answers --


def test_sidecar_does_not_index_commentary_canary(tmp_path: Path) -> None:
    """Plant the fixed canary in commentary and in the spine. After
    rebuild, the canary must not be searchable via the sidecar."""
    ws, _src = _seed_workspace_with_claim(tmp_path)
    # Plant canary in commentary region.
    page_path = ws.pages / "concepts" / "complementarity.md"
    text = page_path.read_text(encoding="utf-8")
    poisoned = text.replace(
        "Human commentary body.",
        f"Human commentary body. {FIXED_CANARY_TOKEN}",
    )
    page_path.write_text(poisoned, encoding="utf-8")
    # Plant canary in spine overview.
    (ws.pages / "overview.md").write_text(
        f"# Overview\n\nSpine {FIXED_CANARY_TOKEN}\n", encoding="utf-8"
    )

    rebuild(ws, target="search")

    # Direct sidecar search for the canary must return nothing.
    hits = search_candidates(ws, FIXED_CANARY_TOKEN)
    assert hits == [], (
        f"canary leaked into sidecar via commentary/spine path: {hits}"
    )

    # And the query answer must not contain the canary either.
    result = query(ws, question=FIXED_CANARY_TOKEN)
    assert FIXED_CANARY_TOKEN not in result.answer


# --- 7. CLI smoke: `llloom rebuild search` creates the sidecar ----------


def test_cli_rebuild_search(tmp_path: Path) -> None:
    ws, _src = _seed_workspace_with_claim(tmp_path)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(["--root", str(ws.root), "rebuild", "search"])
    assert rc == 0

    payload = json.loads(buf.getvalue())
    assert payload["target"] == "search"
    assert payload["claim_rows"] == 1
    assert payload["source_rows"] == 0
    assert ws.search_db.is_file()


# --- 8. FTS5 unavailable: clear error, no sidecar or temp left behind ---


def test_build_search_sidecar_fails_clearly_when_fts5_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the local sqlite3 build lacks FTS5, the rebuild must raise
    `SearchSidecarError` and leave neither the final sidecar nor a
    stray temp database behind."""
    ws, _src = _seed_workspace_with_claim(tmp_path)
    assert not ws.search_db.exists()

    import llloom.state.search as search_module

    def _fail_ensure_fts5(conn: sqlite3.Connection) -> None:  # noqa: ARG001
        raise SearchSidecarError(
            "SQLite FTS5 is not available in this Python's sqlite3 build; "
            "the search sidecar requires FTS5."
        )

    monkeypatch.setattr(search_module, "_ensure_fts5", _fail_ensure_fts5)

    with pytest.raises(SearchSidecarError):
        build_search_sidecar(ws)

    assert not ws.search_db.exists()
    tmp = ws.search_db.with_suffix(ws.search_db.suffix + ".tmp")
    assert not tmp.exists()


# --- 9. rebuild determinism: same counts, no duplicate rows -------------


def test_rebuild_search_is_deterministic(tmp_path: Path) -> None:
    ws, _src = _seed_workspace_with_claim(tmp_path)
    a = rebuild(ws, target="search")
    b = rebuild(ws, target="search")
    assert a["claim_rows"] == b["claim_rows"] == 1
    assert a["source_rows"] == b["source_rows"] == 0

    with closing(sqlite3.connect(str(ws.search_db))) as conn:
        rows = conn.execute(
            "SELECT doc_type, entity_id, claim_id, source_id FROM docs"
        ).fetchall()
    # Exactly one row; no duplicates introduced by the second rebuild.
    assert rows == [("claim", "concept.complementarity", "c.cmp.1", None)]
