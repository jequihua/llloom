"""Rebuildable hybrid search sidecar (SQLite FTS5).

The sidecar lives under ``state/search/search.sqlite`` and is **derived
state only**. It may accelerate candidate selection for ``query``, but
every emitted citation or verbatim span must be rehydrated from the
canonical YAML claim containers or the registered raw source files
before it reaches a ``QueryResult``. Deleting the sidecar must not
cause any data loss.

See ``04_specification/storage_and_state_model.md`` §"Optional later
SQLite sidecars" and ``04_specification/operations_and_cli.md``
§rebuild for the contract.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import yaml

from llloom.claims.store import ClaimStore
from llloom.schema.policy import load_schema
from llloom.sources.registry import SourceRegistry
from llloom.workspace.layout import Workspace


# Claim statuses that are excluded from the search index. Keeping
# retracted / archived / superseded assertions out of the index removes
# a class of stale-hit corner cases; the query-time rehydration path
# also skips these statuses as a defence in depth.
_INACTIVE_CLAIM_STATUSES = frozenset(
    {"retracted", "retracted_by_source", "archived", "superseded", "stale"}
)


class SearchSidecarError(Exception):
    """Raised when the SQLite/FTS5 sidecar cannot be built or opened."""


@dataclass(frozen=True)
class SearchHit:
    """One row returned by the sidecar.

    Never trusted directly: ``query`` re-resolves ``claim`` hits
    through ``ClaimStore``, ``index_only_source`` hits through
    ``SourceRegistry``, and ``structure_item`` hits through the
    report file under ``state/structure/`` before any citation,
    verbatim span, or structure-item record reaches the caller.
    """

    doc_type: str  # "claim" | "index_only_source" | "structure_item"
    entity_id: str | None
    claim_id: str | None
    source_id: str | None
    rank: float
    structure_kind: str | None = None
    structure_name: str | None = None
    structure_symbol_path: str | None = None
    structure_language: str | None = None
    structure_report_path: str | None = None


def sidecar_exists(workspace: Workspace) -> bool:
    return workspace.search_db.is_file()


def _ensure_fts5(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts_probe")
    except sqlite3.OperationalError as exc:
        raise SearchSidecarError(
            "SQLite FTS5 is not available in this Python's sqlite3 build; "
            "the search sidecar requires FTS5. See "
            "https://sqlite.org/fts5.html for build instructions."
        ) from exc


def build_search_sidecar(workspace: Workspace) -> dict:
    """Build (or replace) the FTS5 sidecar at ``workspace.search_db``.

    Builds into a temp file and atomically replaces the old database so
    a failed rebuild leaves no half-built sidecar. Returns a compact
    summary: ``target``, ``index_path``, ``claim_rows``, ``source_rows``.
    """
    workspace.state_search.mkdir(parents=True, exist_ok=True)

    target = workspace.search_db
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    try:
        with closing(sqlite3.connect(str(tmp))) as conn:
            _ensure_fts5(conn)
            _create_schema(conn)
            claim_rows = _index_claims(conn, workspace)
            source_rows = _index_index_only_sources(conn, workspace)
            structure_rows = _index_structure_reports(conn, workspace)
            conn.commit()
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    tmp.replace(target)

    return {
        "target": "search",
        "index_path": target.as_posix(),
        "claim_rows": claim_rows,
        "source_rows": source_rows,
        "structure_rows": structure_rows,
    }


def _create_schema(conn: sqlite3.Connection) -> None:
    # Single FTS5 virtual table. Non-indexed columns carry the doc id
    # metadata that lets rehydration look up canonical records. Keeping
    # the schema flat (no separate metadata table) simplifies the
    # rehydration path without changing the retrieval contract.
    conn.execute(
        """
        CREATE VIRTUAL TABLE docs USING fts5(
            doc_type UNINDEXED,
            entity_id UNINDEXED,
            claim_id UNINDEXED,
            source_id UNINDEXED,
            structure_kind UNINDEXED,
            structure_name UNINDEXED,
            structure_symbol_path UNINDEXED,
            structure_language UNINDEXED,
            structure_report_path UNINDEXED,
            text,
            tokenize = 'unicode61 remove_diacritics 1'
        )
        """
    )


def _index_claims(conn: sqlite3.Connection, workspace: Workspace) -> int:
    store = ClaimStore(workspace)
    registry = SourceRegistry(workspace)
    retracted_sources = {
        rec.source_id for rec in registry.iter_records() if rec.status == "retracted"
    }
    count = 0
    for entity in store.iter_entities():
        for assertion in entity.assertions:
            if assertion.status in _INACTIVE_CLAIM_STATUSES:
                continue
            if any(e.source_id in retracted_sources for e in assertion.evidence):
                # The whole evidence chain must still have at least one
                # non-retracted source for the claim to remain reachable.
                if all(e.source_id in retracted_sources for e in assertion.evidence):
                    continue
            conn.execute(
                "INSERT INTO docs(doc_type, entity_id, claim_id, source_id, "
                "structure_kind, structure_name, structure_symbol_path, "
                "structure_language, structure_report_path, text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "claim",
                    entity.entity_id,
                    assertion.claim_id,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    assertion.claim_text,
                ),
            )
            count += 1
    return count


def _index_index_only_sources(conn: sqlite3.Connection, workspace: Workspace) -> int:
    schema = load_schema(workspace)
    registry = SourceRegistry(workspace)
    count = 0
    for record in registry.iter_records():
        if record.status == "retracted":
            continue
        try:
            policy = schema.resolve_ingest_policy(record.source_class)
        except Exception:
            continue
        if policy != "index_only":
            continue
        try:
            text = registry.raw_text(record)
        except FileNotFoundError:
            continue
        conn.execute(
            "INSERT INTO docs(doc_type, entity_id, claim_id, source_id, "
            "structure_kind, structure_name, structure_symbol_path, "
            "structure_language, structure_report_path, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "index_only_source",
                None,
                None,
                record.source_id,
                None,
                None,
                None,
                None,
                None,
                text,
            ),
        )
        count += 1
    return count


def _index_structure_reports(conn: sqlite3.Connection, workspace: Workspace) -> int:
    """Index derived structure-report items as ``structure_item`` rows.

    The report YAML lives under ``state/structure/<source_id>.yaml``
    and is itself derived state. Rows carry **metadata only**: source
    id, source class, language, item kind, item name, symbol path,
    and the workspace-relative report path. The indexed FTS text is
    assembled from those metadata strings and does not include
    scalar YAML values, comments, docstrings, full source lines,
    code bodies, raw source bodies, or locator-resolved excerpts.

    Stale reports are silently skipped (no crash, no partial row):
    if the source is missing / retracted, if the registered source
    class or content hash no longer matches the report, or if the
    report itself is malformed, the report contributes zero rows.
    """
    state_structure = workspace.state_structure
    if not state_structure.is_dir():
        return 0
    registry = SourceRegistry(workspace)
    records = {rec.source_id: rec for rec in registry.iter_records()}
    count = 0
    for report_path in sorted(state_structure.glob("*.yaml")):
        try:
            data = yaml.safe_load(report_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("version") != "structure_report_v1":
            continue
        source_id = data.get("source_id")
        source_class = data.get("source_class")
        language = data.get("language")
        content_hash = data.get("content_hash")
        items = data.get("items")
        if not (
            isinstance(source_id, str)
            and isinstance(source_class, str)
            and isinstance(language, str)
            and isinstance(content_hash, str)
            and isinstance(items, list)
        ):
            continue
        record = records.get(source_id)
        if record is None or record.status == "retracted":
            continue
        if record.source_class != source_class:
            continue
        if record.content_hash != content_hash:
            continue
        rel_report = _workspace_relative_posix(workspace, report_path)
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            name = item.get("name")
            symbol_path = item.get("symbol_path")
            locator = item.get("locator")
            if not (
                isinstance(kind, str)
                and isinstance(name, str)
                and isinstance(symbol_path, str)
                and isinstance(locator, dict)
            ):
                continue
            text = _structure_item_text(
                source_id=source_id,
                source_class=source_class,
                language=language,
                kind=kind,
                name=name,
                symbol_path=symbol_path,
            )
            conn.execute(
                "INSERT INTO docs(doc_type, entity_id, claim_id, source_id, "
                "structure_kind, structure_name, structure_symbol_path, "
                "structure_language, structure_report_path, text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "structure_item",
                    None,
                    None,
                    source_id,
                    kind,
                    name,
                    symbol_path,
                    language,
                    rel_report,
                    text,
                ),
            )
            count += 1
    return count


def _structure_item_text(
    *,
    source_id: str,
    source_class: str,
    language: str,
    kind: str,
    name: str,
    symbol_path: str,
) -> str:
    """Build the FTS-indexed text for a structure-item row.

    Intentionally narrow: only metadata strings are concatenated.
    A future regression that folded in the item's source body or a
    scalar value would inflate this string and would be caught by
    the "structure rows contain metadata only" contract test.
    """
    return " ".join([source_id, source_class, language, kind, name, symbol_path])


def _workspace_relative_posix(workspace: Workspace, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(workspace.root.resolve())
    except ValueError:
        rel = path
    return rel.as_posix()


def search_candidates(
    workspace: Workspace,
    query_text: str,
    *,
    limit: int = 50,
) -> list[SearchHit]:
    """Return candidate hits from the sidecar.

    Returns an empty list if the sidecar is absent or the query is empty.
    The caller is responsible for rehydrating and revalidating each hit
    against canonical records before trusting it.
    """
    db = workspace.search_db
    if not db.is_file():
        return []
    tokens = _fts_tokens(query_text)
    if not tokens:
        return []
    match_expr = " OR ".join(tokens)
    with closing(sqlite3.connect(str(db))) as conn:
        try:
            cur = conn.execute(
                "SELECT doc_type, entity_id, claim_id, source_id, "
                "structure_kind, structure_name, structure_symbol_path, "
                "structure_language, structure_report_path, rank "
                "FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT ?",
                (match_expr, limit),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []
    return [
        SearchHit(
            doc_type=str(row[0]),
            entity_id=row[1],
            claim_id=row[2],
            source_id=row[3],
            structure_kind=row[4],
            structure_name=row[5],
            structure_symbol_path=row[6],
            structure_language=row[7],
            structure_report_path=row[8],
            rank=float(row[9]) if row[9] is not None else 0.0,
        )
        for row in rows
    ]


def _fts_tokens(text: str) -> list[str]:
    """Extract alnum tokens for an FTS5 MATCH expression.

    FTS5 MATCH treats punctuation as a syntax surface (``.``, ``-``,
    ``"``) so we strip everything non-alphanumeric and re-split. Short
    tokens (< 2 chars) are dropped because they add noise without
    meaningfully helping retrieval in this sidecar.
    """
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    out: list[str] = []
    seen: set[str] = set()
    for tok in cleaned.split():
        if len(tok) < 2:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out
