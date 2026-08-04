"""`render` operation: regenerate claim-block regions from claims.

Slice 071 made rendering page/block-centric: one render call per
page, claim block is the union of every entity's render-visible
assertions targeting `(page_id, block_id)`, ordered by
`(entity_id, claim_id)`. Slice 073 adds a read-only render planning
surface so agents can inspect what render would do before any
write:

- `render(workspace, target=..., dry_run=True)` computes the full
  plan, including would-change flags and planned fingerprints,
  without acquiring the workspace lock, opening a render journal
  entry, or writing any page or fingerprint.
- `render(workspace, target=..., list_targets=True)` enumerates
  valid render targets and their contributors with the same
  no-lock / no-journal / no-write guarantee. A `page:<id>` target
  whose page exists on disk but has no contributing claims still
  surfaces as a plan entry with no contributors and
  ``marker_health="ok"`` instead of silently disappearing.

Target validation runs **lockless** before any workspace mutation
(Slice 068): unknown or missing render targets raise ``ValueError``
before the workspace lock is acquired and before any journal entry
is opened. The same preflight applies on the dry-run / list-targets
paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llloom.claims.models import EntityContainer
from llloom.claims.store import ClaimStore
from llloom.ops._context import operation, relative_posix
from llloom.ops.results import (
    RenderPlanContributor,
    RenderPlanEntry,
    RenderResult as OpsRenderResult,
)
from llloom.pages.regions import PageParseError, parse_page, replace_claim_block
from llloom.pages.render import (
    _RENDER_HIDDEN_STATUSES,
    RenderError,
    compute_render_fingerprint_from_contributors,
    render_claim_block_from_contributors,
    render_page_file_from_contributors,
    resolve_page_path,
)
from llloom.state.fingerprints import FingerprintStore
from llloom.state.render_transactions import RenderTransaction
from llloom.workspace.layout import Workspace


_PAGE_TARGET_PREFIX = "page:"


@dataclass(frozen=True)
class _RenderPlanItem:
    page_id: str
    contributors: tuple[EntityContainer, ...]


def render(
    workspace: Workspace,
    *,
    target: str | None = None,
    dry_run: bool = False,
    list_targets: bool = False,
) -> OpsRenderResult:
    """Render one page (by page_id) or all pages with claim targets.

    ``target`` forms:

    - ``None``: render every page whose page_id appears on any
      assertion's render_targets. Each page is rendered once over
      the union of its contributing entities.
    - ``page:<page_id>``: render only that page. The page must
      either exist on disk under ``pages/`` or be referenced by at
      least one claim's render_targets; otherwise preflight raises
      ``ValueError`` before the workspace lock is acquired.

    ``dry_run`` and ``list_targets`` (Slice 073) flip the operation
    into a read-only inspection mode: target preflight still raises
    on unknown forms, but no workspace lock is acquired, no journal
    entry is opened, no page is written, and no fingerprint is
    written. The returned :class:`OpsRenderResult` carries the
    structured plan on ``result.plan`` with one entry per page,
    including contributors, claim ids, marker health, and
    would-change flags. ``rendered_pages`` / ``unchanged_pages`` /
    ``fingerprints`` stay empty on the dry-run / list-targets path.
    Setting both flags is allowed; they share the same plan output.
    """
    store = ClaimStore(workspace)
    plan = _build_render_plan(workspace, store, target)

    if dry_run or list_targets:
        return _build_inspection_result(
            workspace=workspace,
            store=store,
            target=target,
            base_plan=plan,
            dry_run=dry_run,
            list_targets=list_targets,
        )

    fingerprints = FingerprintStore(workspace)
    with operation(workspace, op_kind="render") as ctx:
        return _commit_under_transaction(
            workspace=workspace,
            plan=plan,
            fingerprints=fingerprints,
            ctx=ctx,
        )


def _commit_under_transaction(
    *,
    workspace: Workspace,
    plan: list[_RenderPlanItem],
    fingerprints: FingerprintStore,
    ctx,
) -> OpsRenderResult:
    """Slice 074: stage every page + the fingerprint snapshot under
    ``state/transactions/<op_id>/``, then atomically commit.

    Order of operations:

    1. Plan: for each item, parse the page and compute new bytes +
       planned union fingerprint. Pages that don't exist on disk
       are skipped (preserving the pre-Slice-074 behavior).
    2. Stage: write the planned page bytes into
       ``state/transactions/<op_id>/pages/...`` and write the full
       planned ``render_fingerprints.yaml`` into
       ``state/transactions/<op_id>/render_fingerprints.yaml``.
       Write a ``manifest.yaml`` describing the staged set.
    3. Commit: replace every changed final page path via
       ``tmp.replace(target)`` (the staged file is the tmp),
       then replace ``state/render_fingerprints.yaml`` with the
       staged snapshot. Update the manifest to ``committed`` and
       remove the transaction directory.

    A pre-commit failure (anything raised before
    :meth:`RenderTransaction.commit` returns) leaves final pages
    and ``state/render_fingerprints.yaml`` byte-identical to
    before the operation, with the transaction directory on disk
    for inspection. The exception propagates through
    ``operation(...)`` which keeps the journal ``in_progress`` and
    the lock held so ``reconcile`` (or an operator) can triage.
    """
    txn = RenderTransaction(workspace, ctx.op_id)
    planned_fingerprints: dict[str, str] = dict(fingerprints.load())
    staged_records: list[tuple[str, Path, str]] = []  # (page_id, path, fingerprint)

    for item in plan:
        page_path = resolve_page_path(workspace, item.page_id)
        if not page_path.is_file():
            continue
        original = page_path.read_text(encoding="utf-8")
        try:
            parsed = parse_page(original)
        except PageParseError as exc:
            raise RenderError(f"{page_path}: {exc}") from exc
        contributors_list = list(item.contributors)
        new_inner = render_claim_block_from_contributors(
            contributors_list, parsed.claim_block_id
        )
        new_text = replace_claim_block(parsed, new_inner)
        fingerprint = compute_render_fingerprint_from_contributors(
            contributors_list, parsed.claim_block_id
        )
        page_id_resolved = str(parsed.frontmatter.get("page_id", item.page_id))
        txn.stage_page(
            page_id=page_id_resolved,
            final_path=page_path,
            original_text=original,
            new_text=new_text,
        )
        planned_fingerprints[page_id_resolved] = fingerprint
        staged_records.append((page_id_resolved, page_path, fingerprint))

    txn.stage_fingerprints(planned_fingerprints)
    txn.write_manifest(status="staged")

    rendered, unchanged = txn.commit()

    out = OpsRenderResult()
    out.rendered_pages.extend(rendered)
    out.unchanged_pages.extend(unchanged)
    for page_id, page_path, fingerprint in staged_records:
        out.fingerprints[page_id] = fingerprint
        rel = relative_posix(workspace, page_path)
        if rel not in ctx.entry.touched_files:
            ctx.entry.touched_files.append(rel)
    return out


def _build_render_plan(
    workspace: Workspace,
    store: ClaimStore,
    target: str | None,
) -> list[_RenderPlanItem]:
    """Lockless preflight: validate ``target`` and collect render items.

    Raises ``ValueError`` with a syntax hint for unknown target forms
    and for ``page:<page_id>`` targets that name neither an on-disk
    page nor a claim-referenced page_id. Never acquires the workspace
    lock; never opens a journal entry. Each plan item carries the
    full contributor set for its page, deterministically sorted by
    ``entity_id``.
    """
    if target is None:
        return _collect_all(store)

    if not target.startswith(_PAGE_TARGET_PREFIX):
        raise ValueError(_unknown_target_message(target))

    page_id = target[len(_PAGE_TARGET_PREFIX) :]
    if not page_id:
        raise ValueError(
            "empty page id in render target 'page:'; "
            "accepted form: 'page:<page_id>' (e.g. 'page:concept/foo')"
        )

    item = _collect_for_page(store, page_id)
    if item is not None:
        return [item]

    if resolve_page_path(workspace, page_id).is_file():
        return []

    raise ValueError(
        f"unknown render target 'page:{page_id}': page is neither "
        f"present on disk under pages/ nor referenced by any claim "
        f"render_targets"
    )


def _collect_all(store: ClaimStore) -> list[_RenderPlanItem]:
    """Group every claim-referenced page with the full contributor set.

    Returns one ``_RenderPlanItem`` per distinct page_id. Pages are
    ordered by page_id; contributors per page are ordered by
    entity_id.
    """
    page_to_contributors: dict[str, list[EntityContainer]] = {}
    for entity in store.iter_entities():
        seen_pages_for_entity: set[str] = set()
        for assertion in entity.assertions:
            for t in assertion.render_targets:
                if t.page_id in seen_pages_for_entity:
                    continue
                seen_pages_for_entity.add(t.page_id)
                page_to_contributors.setdefault(t.page_id, []).append(entity)
    out: list[_RenderPlanItem] = []
    for page_id in sorted(page_to_contributors):
        contributors = page_to_contributors[page_id]
        contributors.sort(key=lambda e: e.entity_id)
        out.append(
            _RenderPlanItem(
                page_id=page_id, contributors=tuple(contributors)
            )
        )
    return out


def _collect_for_page(store: ClaimStore, page_id: str) -> _RenderPlanItem | None:
    """Gather every entity targeting ``page_id`` into one render item.

    Returns ``None`` when no entity targets ``page_id`` (the caller
    decides whether that is a refusal or a valid no-op based on
    on-disk page existence).
    """
    contributors: list[EntityContainer] = []
    for entity in store.iter_entities():
        if any(
            any(t.page_id == page_id for t in a.render_targets)
            for a in entity.assertions
        ):
            contributors.append(entity)
    if not contributors:
        return None
    contributors.sort(key=lambda e: e.entity_id)
    return _RenderPlanItem(page_id=page_id, contributors=tuple(contributors))


def _unknown_target_message(target: str) -> str:
    suggestion = (
        f" did you mean 'page:{target}'?"
        if target and not target.startswith("page:")
        and "/" in target
        and ":" not in target
        else ""
    )
    return (
        f"unknown render target {target!r};{suggestion} "
        f"accepted forms: 'page:<page_id>' or omit the argument to "
        f"render every page referenced by a claim render_target"
    )


# ---- Slice 073: dry-run / list-targets read-only planning -------------


def _build_inspection_result(
    *,
    workspace: Workspace,
    store: ClaimStore,
    target: str | None,
    base_plan: list[_RenderPlanItem],
    dry_run: bool,
    list_targets: bool,
) -> OpsRenderResult:
    """Build a no-write :class:`OpsRenderResult` for dry-run / list.

    Read-only by construction: never enters ``operation(...)``, never
    acquires the workspace lock, never writes a journal entry, never
    writes a page, never writes a fingerprint. The
    ``FingerprintStore`` is consulted via ``load()`` only.
    """
    fps = FingerprintStore(workspace).load()
    plan: list[_RenderPlanItem] = list(base_plan)

    # When the caller passed `page:<id>` for a page that exists on
    # disk with no contributing claims, the real-render plan is
    # empty (silent no-op for the mutating path). For dry-run /
    # list-targets we surface the page anyway so the caller can see
    # it as a valid target with no contributors.
    if (
        not plan
        and target is not None
        and target.startswith(_PAGE_TARGET_PREFIX)
    ):
        page_id = target[len(_PAGE_TARGET_PREFIX) :]
        if resolve_page_path(workspace, page_id).is_file():
            plan = [_RenderPlanItem(page_id=page_id, contributors=())]

    out = OpsRenderResult(dry_run=dry_run, list_targets=list_targets)
    for item in plan:
        out.plan.append(_describe_plan_item(workspace, item, fps))
    return out


def _describe_plan_item(
    workspace: Workspace,
    item: _RenderPlanItem,
    stored_fingerprints: dict[str, str],
) -> RenderPlanEntry:
    """Build one read-only :class:`RenderPlanEntry` for a plan item.

    Computes whether the page content and stored fingerprint would
    change against the current canonical claim state. Surfaces
    parse errors as ``marker_health="parse_error"`` with a message
    instead of raising; the mutating ``render`` path still fails
    hard on the same condition.
    """
    page_path = resolve_page_path(workspace, item.page_id)
    rel_path = relative_posix(workspace, page_path)
    target_str = f"{_PAGE_TARGET_PREFIX}{item.page_id}"

    if not page_path.is_file():
        return RenderPlanEntry(
            target=target_str,
            page_id=item.page_id,
            page_path=rel_path,
            contributors=_describe_contributors_unparsed(item.contributors),
            contributing_claim_ids=[],
            marker_health="missing_page",
            marker_message=(
                f"no page file at {rel_path}; render would skip this page"
            ),
            stored_fingerprint=stored_fingerprints.get(item.page_id),
        )

    original = page_path.read_text(encoding="utf-8")
    try:
        parsed = parse_page(original)
    except PageParseError as exc:
        return RenderPlanEntry(
            target=target_str,
            page_id=item.page_id,
            page_path=rel_path,
            contributors=_describe_contributors_unparsed(item.contributors),
            contributing_claim_ids=[],
            marker_health="parse_error",
            marker_message=str(exc),
            stored_fingerprint=stored_fingerprints.get(item.page_id),
        )

    contributors_list = list(item.contributors)
    block_id = parsed.claim_block_id
    new_inner = render_claim_block_from_contributors(contributors_list, block_id)
    new_text = replace_claim_block(parsed, new_inner)
    planned_fp = compute_render_fingerprint_from_contributors(
        contributors_list, block_id
    )
    page_id_resolved = str(parsed.frontmatter.get("page_id", item.page_id))
    stored_fp = stored_fingerprints.get(page_id_resolved)

    contributor_records = _describe_contributors_for_block(
        contributors_list, block_id
    )
    contributing_claim_ids: list[str] = [
        cid for c in contributor_records for cid in c.claim_ids
    ]

    return RenderPlanEntry(
        target=target_str,
        page_id=page_id_resolved,
        page_path=rel_path,
        block_id=block_id,
        contributors=contributor_records,
        contributing_claim_ids=contributing_claim_ids,
        marker_health="ok",
        marker_message=None,
        content_would_change=(new_text != original),
        fingerprint_would_change=(stored_fp != planned_fp),
        planned_fingerprint=planned_fp,
        stored_fingerprint=stored_fp,
    )


def _describe_contributors_for_block(
    contributors: list[EntityContainer], block_id: str
) -> list[RenderPlanContributor]:
    """Render-visible filter + ``(entity_id, claim_id)`` sort.

    Mirrors :func:`llloom.pages.render._select_contributors` for the
    purpose of plan reporting: entities with no render-visible
    assertion targeting ``block_id`` are dropped; surviving
    contributors carry their claim ids sorted by ``claim_id``; the
    list itself is sorted by ``entity_id``.
    """
    out: list[RenderPlanContributor] = []
    for entity in contributors:
        visible = [
            a
            for a in entity.assertions
            if any(t.block_id == block_id for t in a.render_targets)
            and a.status not in _RENDER_HIDDEN_STATUSES
        ]
        if not visible:
            continue
        visible.sort(key=lambda a: a.claim_id)
        out.append(
            RenderPlanContributor(
                entity_id=entity.entity_id,
                display_name=entity.display_name,
                claim_ids=[a.claim_id for a in visible],
            )
        )
    out.sort(key=lambda c: c.entity_id)
    return out


def _describe_contributors_unparsed(
    contributors: tuple[EntityContainer, ...],
) -> list[RenderPlanContributor]:
    """Best-effort contributor list when the page block id is unknown.

    Used on the ``missing_page`` and ``parse_error`` paths where the
    parsed block id is not available. Returns one entry per
    contributing entity with no claim ids (the render-visible filter
    needs a block id and would over-approximate without one).
    """
    out = [
        RenderPlanContributor(
            entity_id=entity.entity_id,
            display_name=entity.display_name,
            claim_ids=[],
        )
        for entity in contributors
    ]
    out.sort(key=lambda c: c.entity_id)
    return out
