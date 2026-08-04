"""`rebuild` operation.

First-slice rebuild targets:

- ``render_fingerprints``: recompute from current claims
- ``health_report``: regenerate a health summary
- ``index``: regenerate a deterministic index page listing entities
- ``log``: regenerate a log page listing completed journal entries

All rebuild outputs are derived files; they are refused as direct-edit
targets by LLMInvoke.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import yaml

from llloom.claims.store import ClaimStore
from llloom.ops._context import iso_now, operation, relative_posix
from llloom.ops.results import HealthReport
from llloom.pages.render import compute_page_render_fingerprints
from llloom.schema.policy import SchemaError, load_schema
from llloom.sources.registry import SourceRegistry
from llloom.state.fingerprints import FingerprintStore
from llloom.state.journal import JournalEntry, OperationJournal
from llloom.state.lock import LockError, WorkspaceLock
from llloom.state.graph import build_graph_sidecar
from llloom.state.search import build_search_sidecar
from llloom.workspace.layout import Workspace


REBUILD_TARGETS = frozenset(
    {"render_fingerprints", "health_report", "index", "log", "search", "graph"}
)


def rebuild(workspace: Workspace, *, target: str) -> Any:
    if target not in REBUILD_TARGETS:
        raise ValueError(
            f"unknown rebuild target {target!r}; allowed: {sorted(REBUILD_TARGETS)}"
        )
    if target == "health_report":
        # Read-only diagnostic over existing workspace state. Must remain
        # callable when the workspace lock is stale-unrecoverable, since
        # that very condition is one of the drift signals it reports.
        # The function writes its own completed journal entry directly,
        # for audit parity with other rebuild targets, without acquiring
        # the workspace lock.
        return _rebuild_health_report(workspace)
    with operation(workspace, op_kind=f"rebuild.{target}") as ctx:
        if target == "render_fingerprints":
            return _rebuild_render_fingerprints(workspace, ctx)
        if target == "index":
            return _rebuild_index(workspace, ctx)
        if target == "log":
            return _rebuild_log(workspace, ctx)
        if target == "search":
            return _rebuild_search(workspace, ctx)
        if target == "graph":
            return _rebuild_graph(workspace, ctx)
        raise AssertionError("unreachable")  # pragma: no cover


def _rebuild_render_fingerprints(workspace: Workspace, ctx) -> dict:
    store = ClaimStore(workspace)
    fps = FingerprintStore(workspace)
    # Slice 071: page/block-centric union fingerprints. The stored
    # fingerprint for a page is the hash over every contributing
    # entity's render-visible assertions, not any one entity's
    # contribution.
    computed = compute_page_render_fingerprints(store.iter_entities())
    fps.save(computed)
    ctx.entry.touched_files.append(
        relative_posix(workspace, workspace.render_fingerprints)
    )
    return {"target": "render_fingerprints", "fingerprints": computed}


def _rebuild_health_report(workspace: Workspace) -> HealthReport:
    """Read-only drift detection over current workspace state.

    Detection only: never mutates canonical state, never rebuilds
    missing sidecars, never clears locks, never marks journals
    interrupted. Repair belongs to ``reconcile`` and the per-target
    ``rebuild`` verbs.

    Does not acquire the workspace lock. The function writes its own
    completed journal entry for audit parity. This means
    ``rebuild health_report`` can be run safely against a workspace
    that is locked (including the stale-unrecoverable case it must
    report on); it never races canonical state.
    """
    started_at = iso_now()
    store = ClaimStore(workspace)
    registry = SourceRegistry(workspace)
    journal = OperationJournal(workspace)
    lock = WorkspaceLock(workspace)
    fps = FingerprintStore(workspace)

    entity_count = len(store.list_entity_ids())
    claim_count = 0
    stale_claims = 0
    retracted_claims = 0
    for entity in store.iter_entities():
        for assertion in entity.assertions:
            claim_count += 1
            if assertion.status == "stale":
                stale_claims += 1
            if assertion.status in {"retracted", "retracted_by_source"}:
                retracted_claims += 1

    pending_proposals = sum(
        1
        for pid in store.list_proposal_ids()
        if store.load_proposal(pid).status == "pending"
    )

    malformed_lock = False
    current_lock = None
    try:
        current_lock = lock.read()
    except LockError:
        # A malformed lock file is itself a drift signal. Surface it as
        # held-with-no-owner: ``lock_held=True`` so the report does not
        # silently downgrade a corrupt-on-disk lock to "no lock at all",
        # and ``lock_owner=None`` because the on-disk owner is not
        # trustworthy. ``stale_lock_unrecoverable`` stays ``False`` here
        # because the predicate requires a parseable ``Lock`` object
        # to evaluate; the malformed flag is the actionable signal.
        malformed_lock = True
        current_lock = None
    lock_held = malformed_lock or current_lock is not None
    lock_owner = current_lock.owner_id if current_lock is not None else None

    interrupted_journals = _interrupted_journal_op_ids(journal, current_lock)
    stale_lock_unrecoverable = _stale_lock_unrecoverable(
        lock, current_lock, journal
    )

    search_sidecar = "present" if workspace.search_db.is_file() else "missing"
    graph_sidecar = "present" if workspace.graph_db.is_file() else "missing"

    missing_structure_reports, structure_report_drift = (
        _structure_report_drift(workspace, registry)
    )

    render_fingerprint_drift = _render_fingerprint_drift(store, fps)

    report = HealthReport(
        target="health_report",
        entity_count=entity_count,
        claim_count=claim_count,
        pending_proposals=pending_proposals,
        stale_claims=stale_claims,
        retracted_claims=retracted_claims,
        lock_held=lock_held,
        lock_owner=lock_owner,
        interrupted_journals=interrupted_journals,
        stale_lock_unrecoverable=stale_lock_unrecoverable,
        search_sidecar=search_sidecar,
        graph_sidecar=graph_sidecar,
        structure_report_drift=structure_report_drift,
        missing_structure_reports=missing_structure_reports,
        render_fingerprint_drift=render_fingerprint_drift,
    )

    path = workspace.state_reports_health / "health_report.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(asdict(report), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    tmp.replace(path)

    op_id = journal.new_op_id("rebuild.health_report")
    completed = JournalEntry(
        op_id=op_id,
        op_kind="rebuild.health_report",
        status="completed",
        started_at=started_at,
        completed_at=iso_now(),
        lock_id="",
        touched_files=[relative_posix(workspace, path)],
    )
    journal.save(completed)
    return report


def _interrupted_journal_op_ids(
    journal: OperationJournal,
    current_lock,
) -> list[str]:
    held_op_id = current_lock.op_id if current_lock is not None else None
    out: list[str] = []
    for entry in journal.iter_entries():
        if entry.status != "in_progress":
            continue
        if held_op_id is not None and entry.op_id == held_op_id:
            continue
        out.append(entry.op_id)
    return sorted(out)


def _stale_lock_unrecoverable(
    lock: WorkspaceLock,
    current_lock,
    journal: OperationJournal,
) -> bool:
    if current_lock is None:
        return False
    if not lock.is_timed_out(current_lock):
        return False
    recoverable, _ = lock.is_stale_recoverable(current_lock, journal=journal)
    return not recoverable


def _structure_report_drift(
    workspace: Workspace,
    registry: SourceRegistry,
) -> tuple[list[str], list[str]]:
    """Walk registered structured sources and bin them into
    (missing_reports, stale_reports). Retracted sources are skipped.
    Schema-load failure short-circuits to empty lists rather than
    crashing the report.
    """
    try:
        schema = load_schema(workspace)
    except SchemaError:
        return [], []
    missing: list[str] = []
    drift: list[str] = []
    for record in registry.iter_records():
        if record.status == "retracted":
            continue
        try:
            policy = schema.resolve_ingest_policy(record.source_class)
        except SchemaError:
            continue
        if policy != "structure_extract":
            continue
        report_path = workspace.structure_report_path(record.source_id)
        if not report_path.is_file():
            missing.append(record.source_id)
            continue
        if not _structure_report_matches(report_path, record):
            drift.append(record.source_id)
    return sorted(missing), sorted(drift)


def _structure_report_matches(report_path, record) -> bool:
    try:
        data = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("version") != "structure_report_v1":
        return False
    if data.get("source_class") != record.source_class:
        return False
    if data.get("content_hash") != record.content_hash:
        return False
    return True


def _render_fingerprint_drift(
    store: ClaimStore,
    fps: FingerprintStore,
) -> list[str]:
    """Return sorted page_ids whose stored vs recomputed fingerprint
    differs. Treats missing-stored, missing-recomputed, and differing
    values uniformly as drift.
    """
    stored = fps.load()
    # Slice 071: page/block-centric union fingerprints (matches what
    # ``render`` and ``rebuild render_fingerprints`` write).
    recomputed = compute_page_render_fingerprints(store.iter_entities())
    drifted: set[str] = set()
    for page_id, fp in stored.items():
        if page_id not in recomputed or recomputed[page_id] != fp:
            drifted.add(page_id)
    for page_id in recomputed:
        if page_id not in stored:
            drifted.add(page_id)
    return sorted(drifted)


def _rebuild_index(workspace: Workspace, ctx) -> dict:
    store = ClaimStore(workspace)
    lines = ["# Index", ""]
    for entity_id in store.list_entity_ids():
        entity = store.load_entity(entity_id)
        lines.append(f"- {entity.display_name} ({entity.entity_type}) â€” `{entity.entity_id}`")
    text = "\n".join(lines) + "\n"
    path = workspace.state_rebuild / "index.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": [
            {
                "entity_id": e,
                "display_name": store.load_entity(e).display_name,
                "entity_type": store.load_entity(e).entity_type,
            }
            for e in store.list_entity_ids()
        ]
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    tmp.replace(path)
    ctx.entry.touched_files.append(relative_posix(workspace, path))
    return {"target": "index", "entries": payload["entries"]}


def _rebuild_search(workspace: Workspace, ctx) -> dict:
    summary = build_search_sidecar(workspace)
    ctx.entry.touched_files.append(
        relative_posix(workspace, workspace.search_db)
    )
    return summary


def _rebuild_graph(workspace: Workspace, ctx) -> dict:
    summary = build_graph_sidecar(workspace)
    ctx.entry.touched_files.append(
        relative_posix(workspace, workspace.graph_db)
    )
    return summary


def _rebuild_log(workspace: Workspace, ctx) -> dict:
    journal = OperationJournal(workspace)
    entries = []
    for entry in journal.iter_entries():
        entries.append(
            {
                "op_id": entry.op_id,
                "op_kind": entry.op_kind,
                "status": entry.status,
                "started_at": entry.started_at,
                "completed_at": entry.completed_at,
            }
        )
    path = workspace.state_rebuild / "log.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump({"entries": entries}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    tmp.replace(path)
    ctx.entry.touched_files.append(relative_posix(workspace, path))
    return {"target": "log", "entries": entries}

