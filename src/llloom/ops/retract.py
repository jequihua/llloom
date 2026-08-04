"""`retract` operation.

Retraction cascade rules from
``04_specification/operations_and_cli.md`` Â§retract:

- sole-supported draft claim -> retracted_by_source
- sole-supported reviewed/validated claim -> stale
- rendered pages with affected claims -> stale_render (recorded via
  rerender plan)

Leaves an inspectable tombstone trail under ``state/reports/health/``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from llloom.claims.store import ClaimStore
from llloom.ops._context import iso_now, operation, relative_posix
from llloom.ops.results import RetractResult
from llloom.claims.models import EntityContainer
from llloom.pages.render import render_page_file_from_contributors, resolve_page_path
from llloom.sources.registry import SourceRegistry
from llloom.state.fingerprints import FingerprintStore
from llloom.workspace.layout import Workspace


def retract(
    workspace: Workspace,
    *,
    source_id: str,
    reason: str | None = None,
) -> RetractResult:
    registry = SourceRegistry(workspace)
    store = ClaimStore(workspace)
    fingerprints = FingerprintStore(workspace)

    with operation(workspace, op_kind="retract") as ctx:
        registry.mark_retracted(source_id, reason=reason)

        affected_claims: list[str] = []
        affected_pages: set[str] = set()
        for entity in store.iter_entities():
            changed = False
            for assertion in entity.assertions:
                cites_source = [e for e in assertion.evidence if e.source_id == source_id]
                if not cites_source:
                    continue
                sole_supported = all(e.source_id == source_id for e in assertion.evidence)
                if sole_supported:
                    if assertion.status == "draft":
                        assertion.status = "retracted_by_source"
                    elif assertion.status in {"reviewed", "validated"}:
                        assertion.status = "stale"
                    # other statuses are left alone
                    assertion.updated_at = iso_now()
                    changed = True
                    affected_claims.append(assertion.claim_id)
                    for t in assertion.render_targets:
                        affected_pages.add(t.page_id)
                else:
                    # Keep claim but annotate verification; still mark a pending review.
                    assertion.updated_at = iso_now()
                    changed = True
                    affected_claims.append(assertion.claim_id)
            if changed:
                store.save_entity(entity)
                ctx.entry.touched_files.append(
                    relative_posix(workspace, store.entity_path(entity.entity_id))
                )

        rerendered: list[str] = []
        affected_page_paths: set[str] = set()
        for page_id in sorted(affected_pages):
            page_path = resolve_page_path(workspace, page_id)
            rel_path = (
                relative_posix(workspace, page_path)
                if page_path.is_file()
                else f"pages/{page_id}.md"
            )
            affected_page_paths.add(rel_path)
            # Slice 071: render the page once over the union of every
            # entity that targets it. The retract cascade no longer
            # writes per-entity claim blocks (which previously caused
            # the last entity to overwrite earlier contributors).
            if not page_path.is_file():
                continue
            contributors: list[EntityContainer] = []
            for entity in store.iter_entities():
                if any(
                    any(t.page_id == page_id for t in a.render_targets)
                    for a in entity.assertions
                ):
                    contributors.append(entity)
            contributors.sort(key=lambda e: e.entity_id)
            render_result = render_page_file_from_contributors(
                workspace, page_path, contributors
            )
            fingerprints.set(render_result.page_id, render_result.fingerprint)
            rerendered.append(relative_posix(workspace, page_path))

        tombstone_path = _write_tombstone(
            workspace=workspace,
            source_id=source_id,
            reason=reason,
            affected_claim_ids=affected_claims,
            affected_pages=sorted(affected_page_paths),
        )
        ctx.entry.touched_files.append(relative_posix(workspace, tombstone_path))
        return RetractResult(
            source_id=source_id,
            tombstone_path=relative_posix(workspace, tombstone_path),
            affected_claim_ids=affected_claims,
            affected_pages=sorted(affected_page_paths),
            rerendered_pages=rerendered,
            op_id=ctx.op_id,
        )


def _write_tombstone(
    *,
    workspace: Workspace,
    source_id: str,
    reason: str | None,
    affected_claim_ids: list[str],
    affected_pages: list[str],
) -> Path:
    directory = workspace.state_reports_health
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"tombstone.{source_id}.yaml"
    payload = {
        "source_id": source_id,
        "retracted_at": iso_now(),
        "reason": reason,
        "affected_claim_ids": affected_claim_ids,
        "affected_pages": affected_pages,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path

