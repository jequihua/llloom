"""Rebuildable graph sidecar (SQLite).

The sidecar lives under ``state/graph/graph.sqlite`` and is **derived
state only**. It may accelerate neighbor lookup over canonical
``relations:`` records in entity YAML containers, but every returned
``GraphEdge`` is rehydrated from canonical YAML and revalidated
before it reaches the caller. Deleting the sidecar must not cause
any data loss.

See ``04_specification/storage_and_state_model.md`` §"Optional later
SQLite sidecars" and ``04_specification/operations_and_cli.md``
§rebuild for the contract.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llloom.claims.store import ClaimStore
from llloom.sources.registry import SourceRegistry
from llloom.workspace.layout import Workspace


_STRUCTURE_REPORT_VERSION = "structure_report_v1"


# Claim statuses that disqualify a claim from being a live graph
# endpoint. An edge whose source or target claim is inactive is
# filtered both at build time and at neighbor-lookup time.
_INACTIVE_CLAIM_STATUSES = frozenset(
    {"retracted", "retracted_by_source", "archived", "superseded", "stale"}
)

# Relation statuses that are excluded from the graph. Only explicitly
# active relations contribute edges. Any unrecognized / non-"active"
# status is treated as inactive; this is the stronger contract.
_ACTIVE_RELATION_STATUS = "active"

_ALLOWED_DIRECTIONS = frozenset({"in", "out", "both"})


class GraphSidecarError(Exception):
    """Raised when the SQLite graph sidecar cannot be built or when a
    neighbor query is given unsupported inputs."""


@dataclass(frozen=True)
class GraphEdge:
    """One rehydrated relation edge.

    Always produced from canonical YAML via ``graph_neighbors``; never
    trusted directly from SQLite. ``source_entity_id`` is the entity
    whose YAML container holds the relation record, which is also the
    entity that owns ``source_claim_id``. ``target_entity_id`` is the
    entity whose YAML container holds ``target_claim_id``.
    """

    relation_id: str
    source_entity_id: str
    source_claim_id: str
    relation_type: str
    target_entity_id: str
    target_claim_id: str
    status: str


def graph_sidecar_exists(workspace: Workspace) -> bool:
    return workspace.graph_db.is_file()


def build_graph_sidecar(workspace: Workspace) -> dict:
    """Build (or replace) the graph sidecar at ``workspace.graph_db``.

    Builds into a temp file and replaces the old database directly so
    a failed rebuild never leaves a half-built sidecar behind and does
    not briefly remove the last good sidecar. Returns a compact
    summary: ``target``, ``index_path``, ``edge_rows``,
    ``structure_edge_rows``.

    The sidecar carries two derived tables. ``edges`` indexes active
    claim-relation edges (unchanged). ``structure_edges`` indexes
    direct parent/child containment edges over derived
    structure reports (additive). Both are non-canonical and may be
    deleted without data loss; every public emission is rehydrated
    from canonical sources (entity YAML for claim relations;
    on-disk structure report for structure edges) before reaching a
    caller.
    """
    workspace.state_graph.mkdir(parents=True, exist_ok=True)

    target = workspace.graph_db
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    try:
        with closing(sqlite3.connect(str(tmp))) as conn:
            _create_schema(conn)
            edge_rows = _index_edges(conn, workspace)
            structure_edge_rows = _index_structure_edges(conn, workspace)
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
        "target": "graph",
        "index_path": target.as_posix(),
        "edge_rows": edge_rows,
        "structure_edge_rows": structure_edge_rows,
    }


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE edges (
            relation_id TEXT PRIMARY KEY,
            source_entity_id TEXT NOT NULL,
            source_claim_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            target_claim_id TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_edges_source_claim ON edges(source_claim_id)")
    conn.execute("CREATE INDEX idx_edges_target_claim ON edges(target_claim_id)")
    conn.execute("CREATE INDEX idx_edges_type ON edges(relation_type)")

    # Structure-report containment edges. Metadata-only; the report
    # under ``state/structure/<source_id>.yaml`` is the source of truth
    # and every public emission is rehydrated from there.
    conn.execute(
        """
        CREATE TABLE structure_edges (
            source_id TEXT NOT NULL,
            parent_symbol_path TEXT NOT NULL,
            child_symbol_path TEXT NOT NULL,
            child_kind TEXT NOT NULL,
            child_name TEXT NOT NULL,
            report_path TEXT NOT NULL,
            PRIMARY KEY (source_id, parent_symbol_path, child_symbol_path)
        )
        """
    )
    conn.execute(
        "CREATE INDEX idx_structure_edges_parent "
        "ON structure_edges(source_id, parent_symbol_path)"
    )
    conn.execute(
        "CREATE INDEX idx_structure_edges_child "
        "ON structure_edges(source_id, child_symbol_path)"
    )


def _build_claim_index(workspace: Workspace) -> dict[str, tuple[str, str]]:
    """Map ``claim_id`` to ``(entity_id, status)`` across the workspace."""
    index: dict[str, tuple[str, str]] = {}
    store = ClaimStore(workspace)
    for entity in store.iter_entities():
        for assertion in entity.assertions:
            index[assertion.claim_id] = (entity.entity_id, assertion.status)
    return index


def _index_edges(conn: sqlite3.Connection, workspace: Workspace) -> int:
    store = ClaimStore(workspace)
    claim_index = _build_claim_index(workspace)
    seen_ids: set[str] = set()
    count = 0
    for entity in store.iter_entities():
        for relation in entity.relations:
            if relation.relation_id in seen_ids:
                continue
            if relation.status != _ACTIVE_RELATION_STATUS:
                continue
            src = claim_index.get(relation.source_claim_id)
            if src is None or src[0] != entity.entity_id:
                # The owning entity must contain the source claim; a
                # relation whose source_claim_id lives elsewhere is a
                # schema violation and is skipped.
                continue
            if src[1] in _INACTIVE_CLAIM_STATUSES:
                continue
            tgt = claim_index.get(relation.target_claim_id)
            if tgt is None:
                continue
            if tgt[1] in _INACTIVE_CLAIM_STATUSES:
                continue
            conn.execute(
                "INSERT INTO edges(relation_id, source_entity_id, source_claim_id, "
                "relation_type, target_entity_id, target_claim_id, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    relation.relation_id,
                    entity.entity_id,
                    relation.source_claim_id,
                    relation.relation_type,
                    tgt[0],
                    relation.target_claim_id,
                    relation.status,
                ),
            )
            seen_ids.add(relation.relation_id)
            count += 1
    return count


def graph_neighbors(
    workspace: Workspace,
    *,
    claim_id: str,
    direction: str = "both",
    relation_types: set[str] | None = None,
    include_inactive: bool = False,
    limit: int = 50,
) -> list[GraphEdge]:
    """Return rehydrated neighbor edges for ``claim_id``.

    The sidecar (if present) is used only to narrow the candidate
    relation set. Every returned edge is rebuilt from canonical YAML
    and revalidated: the owning entity must still contain the
    relation, both endpoint claims must exist and be active (unless
    ``include_inactive`` is True), and the relation must still be
    active (unless ``include_inactive`` is True).

    Without a sidecar, the function falls back to a full canonical
    scan and still returns correct edges; this proves the sidecar is
    an accelerator, not an authority.
    """
    if direction not in _ALLOWED_DIRECTIONS:
        raise GraphSidecarError(
            f"unknown direction {direction!r}; allowed: {sorted(_ALLOWED_DIRECTIONS)}"
        )
    if limit <= 0:
        return []

    candidate_relation_ids = _candidates_from_sidecar(
        workspace, claim_id=claim_id, direction=direction, limit=limit
    )

    store = ClaimStore(workspace)
    claim_index = _build_claim_index(workspace)

    edges: list[GraphEdge] = []
    produced: set[str] = set()

    for entity in store.iter_entities():
        for relation in entity.relations:
            if relation.relation_id in produced:
                continue
            if (
                candidate_relation_ids is not None
                and relation.relation_id not in candidate_relation_ids
            ):
                continue
            if not _edge_matches(
                entity_id=entity.entity_id,
                relation=relation,
                claim_id=claim_id,
                direction=direction,
            ):
                continue
            if relation_types is not None and relation.relation_type not in relation_types:
                continue
            if not include_inactive and relation.status != _ACTIVE_RELATION_STATUS:
                continue
            src = claim_index.get(relation.source_claim_id)
            if src is None or src[0] != entity.entity_id:
                continue
            if not include_inactive and src[1] in _INACTIVE_CLAIM_STATUSES:
                continue
            tgt = claim_index.get(relation.target_claim_id)
            if tgt is None:
                continue
            if not include_inactive and tgt[1] in _INACTIVE_CLAIM_STATUSES:
                continue
            edges.append(
                GraphEdge(
                    relation_id=relation.relation_id,
                    source_entity_id=entity.entity_id,
                    source_claim_id=relation.source_claim_id,
                    relation_type=relation.relation_type,
                    target_entity_id=tgt[0],
                    target_claim_id=relation.target_claim_id,
                    status=relation.status,
                )
            )
            produced.add(relation.relation_id)
            if len(edges) >= limit:
                return edges
    return edges


def _edge_matches(
    *,
    entity_id: str,
    relation,
    claim_id: str,
    direction: str,
) -> bool:
    if direction == "out":
        return relation.source_claim_id == claim_id
    if direction == "in":
        return relation.target_claim_id == claim_id
    return relation.source_claim_id == claim_id or relation.target_claim_id == claim_id


def _candidates_from_sidecar(
    workspace: Workspace,
    *,
    claim_id: str,
    direction: str,
    limit: int,
) -> set[str] | None:
    """Return a set of candidate relation ids from the sidecar, or
    ``None`` if no sidecar is present (the caller should then do a
    full canonical scan)."""
    db = workspace.graph_db
    if not db.is_file():
        return None
    if direction == "out":
        sql = "SELECT relation_id FROM edges WHERE source_claim_id = ? LIMIT ?"
        params: tuple = (claim_id, limit)
    elif direction == "in":
        sql = "SELECT relation_id FROM edges WHERE target_claim_id = ? LIMIT ?"
        params = (claim_id, limit)
    else:
        sql = (
            "SELECT relation_id FROM edges "
            "WHERE source_claim_id = ? OR target_claim_id = ? LIMIT ?"
        )
        params = (claim_id, claim_id, limit)
    try:
        with closing(sqlite3.connect(str(db))) as conn:
            cur = conn.execute(sql, params)
            return {str(row[0]) for row in cur.fetchall()}
    except sqlite3.OperationalError:
        return None


# ---------------------------------------------------------------------------
# Structure-report containment edges
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureGraphEdge:
    """One rehydrated direct parent/child containment edge from a
    structure report.

    Always produced from the on-disk report at
    ``state/structure/<source_id>.yaml`` via
    :func:`structure_graph_neighbors`; never trusted directly from
    SQLite. Carries structure metadata only — scalar values,
    comments, docstrings, code bodies, raw source text,
    locator-resolved excerpts, rendered prose, and model output are
    never present.
    """

    source_id: str
    source_class: str
    language: str
    parent_symbol_path: str
    child_symbol_path: str
    child_kind: str
    child_name: str
    report_path: str


def _direct_parent_symbol_path(symbol_path: str) -> str | None:
    """Return the direct parent symbol path, or ``None`` for roots.

    Direct containment only: strip the final dot-delimited segment.
    ``"policies.markdown_prose"`` -> ``"policies"``;
    ``"root.[0].name"`` -> ``"root.[0]"``;
    ``"defaults"`` -> ``None``.
    """
    idx = symbol_path.rfind(".")
    if idx <= 0:
        return None
    return symbol_path[:idx]


def _load_structure_report(
    workspace: Workspace,
    source_id: str,
    *,
    registry_record,
) -> dict[str, Any] | None:
    """Load and validate the report for ``source_id``.

    Returns the parsed mapping iff every defensive gate passes;
    otherwise ``None``. Mirrors the validation chain used by the
    search-sidecar structure-report path so stale or malformed
    reports contribute nothing and never crash.
    """
    report_path = workspace.structure_report_path(source_id)
    if not report_path.is_file():
        return None
    try:
        data = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("version") != _STRUCTURE_REPORT_VERSION:
        return None
    source_class = data.get("source_class")
    language = data.get("language")
    content_hash = data.get("content_hash")
    items = data.get("items")
    if not (
        isinstance(source_class, str)
        and isinstance(language, str)
        and isinstance(content_hash, str)
        and isinstance(items, list)
    ):
        return None
    if registry_record.status == "retracted":
        return None
    if registry_record.source_class != source_class:
        return None
    if registry_record.content_hash != content_hash:
        return None
    return data


def _index_structure_edges(conn: sqlite3.Connection, workspace: Workspace) -> int:
    """Index direct parent/child containment edges per structure report.

    Each report contributes one row per item that has a direct
    parent. Rows carry metadata only. Stale or malformed reports are
    silently skipped — they contribute zero rows; the rebuild never
    crashes on a bad report.
    """
    state_structure = workspace.state_structure
    if not state_structure.is_dir():
        return 0
    registry = SourceRegistry(workspace)
    records = {rec.source_id: rec for rec in registry.iter_records()}
    count = 0
    seen: set[tuple[str, str, str]] = set()
    for report_path in sorted(state_structure.glob("*.yaml")):
        source_id_from_name = report_path.stem
        record = records.get(source_id_from_name)
        if record is None:
            continue
        data = _load_structure_report(
            workspace, source_id_from_name, registry_record=record
        )
        if data is None:
            continue
        # Defensive: the report's own source_id should match the file
        # stem; if it does not, treat the report as malformed and skip.
        if data.get("source_id") != source_id_from_name:
            continue
        rel_report = _workspace_relative_posix(workspace, report_path)
        items = data["items"]
        # Maintain document order: items already appear in the report's
        # deterministic order. We dedupe by (source_id, parent, child)
        # so a report that lists a symbol twice contributes one row.
        symbol_paths = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            symbol_path = item.get("symbol_path")
            if isinstance(symbol_path, str):
                symbol_paths.add(symbol_path)
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            name = item.get("name")
            symbol_path = item.get("symbol_path")
            if not (
                isinstance(kind, str)
                and isinstance(name, str)
                and isinstance(symbol_path, str)
            ):
                continue
            parent = _direct_parent_symbol_path(symbol_path)
            if parent is None:
                continue
            if parent not in symbol_paths:
                # Skip edges to a parent not declared in this report;
                # we never invent containment that the report did not.
                continue
            key = (source_id_from_name, parent, symbol_path)
            if key in seen:
                continue
            seen.add(key)
            conn.execute(
                "INSERT INTO structure_edges("
                "source_id, parent_symbol_path, child_symbol_path, "
                "child_kind, child_name, report_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source_id_from_name,
                    parent,
                    symbol_path,
                    kind,
                    name,
                    rel_report,
                ),
            )
            count += 1
    return count


def _workspace_relative_posix(workspace: Workspace, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(workspace.root.resolve())
    except ValueError:
        return path.as_posix()
    return rel.as_posix()


def structure_graph_neighbors(
    workspace: Workspace,
    *,
    source_id: str,
    symbol_path: str,
    direction: str = "both",
    limit: int = 50,
) -> list[StructureGraphEdge]:
    """Return rehydrated direct parent/child edges for one symbol.

    The sidecar (if present) narrows candidate ``(parent, child)``
    pairs; every emitted edge is rebuilt from the **current** report
    under ``state/structure/<source_id>.yaml`` and revalidated
    against the **current** source registry record
    (``status`` != ``retracted``, matching ``source_class`` and
    ``content_hash``). Stale rows drop silently.

    Without a sidecar, walks the current report directly and still
    returns correct edges — the SQLite path is an accelerator, not
    an authority. Output order follows the report's document order
    so callers see deterministic results regardless of SQLite row
    order. ``direction`` uses the same values as
    :func:`graph_neighbors`.
    """
    if direction not in _ALLOWED_DIRECTIONS:
        raise GraphSidecarError(
            f"unknown direction {direction!r}; allowed: "
            f"{sorted(_ALLOWED_DIRECTIONS)}"
        )
    if limit <= 0:
        return []

    registry = SourceRegistry(workspace)
    try:
        record = registry.load(source_id)
    except Exception:
        return []
    data = _load_structure_report(workspace, source_id, registry_record=record)
    if data is None:
        return []
    if data.get("source_id") != source_id:
        return []
    items = data["items"]
    source_class = data["source_class"]
    language = data["language"]
    report_rel = _workspace_relative_posix(
        workspace, workspace.structure_report_path(source_id)
    )

    declared_paths: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            sp = item.get("symbol_path")
            if isinstance(sp, str):
                declared_paths.add(sp)
    if symbol_path not in declared_paths:
        return []

    candidate_pairs = _structure_candidates_from_sidecar(
        workspace,
        source_id=source_id,
        symbol_path=symbol_path,
        direction=direction,
        limit=limit,
    )

    edges: list[StructureGraphEdge] = []
    produced: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        name = item.get("name")
        child = item.get("symbol_path")
        if not (
            isinstance(kind, str)
            and isinstance(name, str)
            and isinstance(child, str)
        ):
            continue
        parent = _direct_parent_symbol_path(child)
        if parent is None:
            continue
        if parent not in declared_paths:
            continue
        if not _structure_edge_matches(
            parent=parent,
            child=child,
            symbol_path=symbol_path,
            direction=direction,
        ):
            continue
        pair = (parent, child)
        if (
            candidate_pairs is not None
            and pair not in candidate_pairs
        ):
            continue
        if pair in produced:
            continue
        # For ``in`` direction, child kind/name from the item is the
        # current symbol's metadata; we still want to surface it on
        # the edge so callers can recover (parent_symbol_path,
        # child_symbol_path) consistently regardless of direction.
        edges.append(
            StructureGraphEdge(
                source_id=source_id,
                source_class=source_class,
                language=language,
                parent_symbol_path=parent,
                child_symbol_path=child,
                child_kind=kind,
                child_name=name,
                report_path=report_rel,
            )
        )
        produced.add(pair)
        if len(edges) >= limit:
            break
    return edges


def _structure_edge_matches(
    *,
    parent: str,
    child: str,
    symbol_path: str,
    direction: str,
) -> bool:
    if direction == "out":
        return parent == symbol_path
    if direction == "in":
        return child == symbol_path
    return parent == symbol_path or child == symbol_path


def _structure_candidates_from_sidecar(
    workspace: Workspace,
    *,
    source_id: str,
    symbol_path: str,
    direction: str,
    limit: int,
) -> set[tuple[str, str]] | None:
    """Return ``{(parent_symbol_path, child_symbol_path), ...}`` from the
    sidecar, or ``None`` if no sidecar is present. The caller still
    rehydrates from the report; this set is only a filter.
    """
    db = workspace.graph_db
    if not db.is_file():
        return None
    if direction == "out":
        sql = (
            "SELECT parent_symbol_path, child_symbol_path FROM structure_edges "
            "WHERE source_id = ? AND parent_symbol_path = ? LIMIT ?"
        )
        params: tuple = (source_id, symbol_path, limit)
    elif direction == "in":
        sql = (
            "SELECT parent_symbol_path, child_symbol_path FROM structure_edges "
            "WHERE source_id = ? AND child_symbol_path = ? LIMIT ?"
        )
        params = (source_id, symbol_path, limit)
    else:
        sql = (
            "SELECT parent_symbol_path, child_symbol_path FROM structure_edges "
            "WHERE source_id = ? "
            "AND (parent_symbol_path = ? OR child_symbol_path = ?) LIMIT ?"
        )
        params = (source_id, symbol_path, symbol_path, limit)
    try:
        with closing(sqlite3.connect(str(db))) as conn:
            cur = conn.execute(sql, params)
            return {(str(row[0]), str(row[1])) for row in cur.fetchall()}
    except sqlite3.OperationalError:
        return None
