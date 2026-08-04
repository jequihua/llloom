"""`reconcile` operation.

Frozen state table from ``04_specification/operations_and_cli.md`` Â§reconcile:

- no lock + no journal => no-op
- live lock (not stale) + journal in_progress => refuse
- stale lock + journal in_progress + no partial files => mark journal
  interrupted, clear lock
- stale lock + journal in_progress + orphan temp claim files => delete
  temp files, mark journal interrupted, clear lock
- stale lock + journal in_progress + partial render output => mark
  journal interrupted, clear lock, rerender from current claims
- no lock + completed journal + stale render fingerprint => rerender
  from current claims

Claim writes are atomic; partial claim files are deleted, never
resumed.

Slice 071: the post-cleanup re-render walks pages, not entities. For
each affected page, the full contributor set is gathered from the
canonical store and rendered once over the union — matching the
fingerprint that ``render`` and ``ingest`` now write.
"""

from __future__ import annotations

from pathlib import Path

from llloom.claims.models import EntityContainer
from llloom.claims.store import ClaimStore
from llloom.ops._context import relative_posix
from llloom.ops.results import ReconcileResult
from llloom.pages.regions import PageParseError, parse_page
from llloom.pages.render import (
    compute_render_fingerprint_from_contributors,
    render_page_file_from_contributors,
    resolve_page_path,
)
from llloom.state.fingerprints import FingerprintStore
from llloom.state.journal import OperationJournal
from llloom.state.lock import WorkspaceLock
from llloom.workspace.layout import Workspace


def reconcile(workspace: Workspace) -> ReconcileResult:
    result = ReconcileResult()
    lock = WorkspaceLock(workspace)
    journal = OperationJournal(workspace)
    store = ClaimStore(workspace)
    fingerprints = FingerprintStore(workspace)

    current_lock = lock.read()
    if current_lock is not None:
        if not lock.is_timed_out(current_lock):
            result.actions.append(
                f"refuse: workspace lock held by live op_id={current_lock.op_id}"
            )
            return result
        # Lock has timed out. Apply the frozen journal-backed rule.
        recoverable, reason = lock.is_stale_recoverable(
            current_lock, journal=journal
        )
        if not recoverable:
            # Timed-out but not recoverable per the spec rule. Report and
            # leave the lock untouched so a human can investigate.
            result.actions.append(
                f"refuse: timed-out lock op_id={current_lock.op_id} not recoverable: "
                f"{reason}"
            )
            return result
        op_id = current_lock.op_id
        # Recoverable path: clear orphan temp files, mark journal
        # interrupted, then clear the lock.
        removed = _remove_temp_files(workspace)
        result.temp_files_removed.extend(removed)
        journal.mark_interrupted(op_id, note="reconcile: stale lock cleared")
        result.journals_marked_interrupted.append(op_id)
        lock.clear()
        result.lock_cleared = True
        result.actions.append(f"cleared stale lock op_id={op_id}")
        # Slice 074: an abandoned render transaction directory for the
        # same op_id is in-flight write state that the cleared op will
        # never resume. The journal-backed stale-recovery predicate
        # already proved it is safe to drop. Read-only access only:
        # never touch transaction directories whose op_id does not
        # match the stale lock we just cleared.
        txn_dir = workspace.state_transactions / op_id
        if txn_dir.is_dir():
            import shutil

            shutil.rmtree(txn_dir, ignore_errors=True)
            result.actions.append(
                f"removed abandoned render transaction directory op_id={op_id}"
            )

    # Rerender any pages whose stored fingerprint disagrees with the
    # union fingerprint over the current contributor set.
    fps = fingerprints.load()
    page_to_contributors: dict[str, list[EntityContainer]] = {}
    for entity in store.iter_entities():
        seen_pages_for_entity: set[str] = set()
        for assertion in entity.assertions:
            for target in assertion.render_targets:
                if target.page_id in seen_pages_for_entity:
                    continue
                seen_pages_for_entity.add(target.page_id)
                page_to_contributors.setdefault(target.page_id, []).append(
                    entity
                )
    rerendered: list[str] = []
    for page_id in sorted(page_to_contributors):
        contributors = page_to_contributors[page_id]
        contributors.sort(key=lambda e: e.entity_id)
        page_path = resolve_page_path(workspace, page_id)
        if not page_path.is_file():
            continue
        try:
            parsed = parse_page(page_path.read_text(encoding="utf-8"))
        except PageParseError:
            # A malformed page is not silently re-rendered here; the
            # render op (and lint) surface the parse failure on demand.
            continue
        expected = compute_render_fingerprint_from_contributors(
            contributors, parsed.claim_block_id
        )
        actual = fps.get(page_id)
        if actual is None or actual != expected:
            rendered = render_page_file_from_contributors(
                workspace, page_path, contributors
            )
            fingerprints.set(rendered.page_id, rendered.fingerprint)
            rerendered.append(relative_posix(workspace, page_path))
    result.pages_rerendered.extend(rerendered)

    return result


def _remove_temp_files(workspace: Workspace) -> list[str]:
    removed: list[str] = []
    for root in (workspace.claims, workspace.pages, workspace.state):
        if not root.is_dir():
            continue
        for tmp in root.rglob("*.tmp"):
            try:
                tmp.unlink()
                removed.append(relative_posix(workspace, tmp))
            except FileNotFoundError:
                pass
    return removed
