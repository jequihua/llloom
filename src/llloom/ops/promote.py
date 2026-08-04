"""`promote` operation.

Allowed transitions (first slice):

- draft -> reviewed
- reviewed -> validated
- validated -> superseded
- validated -> archived

Promotion refuses if verification did not pass.

Slice 078 centralized the legal transition graph in
:mod:`llloom.claims.lifecycle`. ``ALLOWED_TRANSITIONS`` is now a
back-compat alias for ``LEGAL_LIFECYCLE_TRANSITIONS`` so any
out-of-tree caller that imported the constant keeps working, but
``promote(...)`` itself consults the shared
:func:`can_transition` / :func:`explain_transition_refusal`
helpers — the supersede / lifecycle helpers and ``promote`` now
share a single source of truth.
"""

from __future__ import annotations

from llloom.claims.lifecycle import (
    LEGAL_LIFECYCLE_TRANSITIONS,
    can_transition,
    explain_transition_refusal,
)
from llloom.claims.store import ClaimStore
from llloom.claims.verifier import verify_assertion
from llloom.ops._context import iso_now, operation, relative_posix
from llloom.ops.results import PromoteResult
from llloom.sources.registry import SourceRegistry
from llloom.workspace.layout import Workspace


# Back-compat alias. Slice 078's
# ``llloom.claims.lifecycle.LEGAL_LIFECYCLE_TRANSITIONS`` is the
# new canonical source of truth.
ALLOWED_TRANSITIONS = LEGAL_LIFECYCLE_TRANSITIONS


def promote(
    workspace: Workspace,
    *,
    target: str,  # "claim:<entity_id>:<claim_id>"
    to_status: str,
) -> PromoteResult:
    if not target.startswith("claim:"):
        raise ValueError(
            f"promote target must be 'claim:<entity_id>:<claim_id>', got {target!r}"
        )
    _, rest = target.split(":", 1)
    entity_id, claim_id = rest.split(":", 1)

    store = ClaimStore(workspace)
    registry = SourceRegistry(workspace)

    with operation(workspace, op_kind="promote") as ctx:
        entity = store.load_entity(entity_id)
        assertion = entity.find_assertion(claim_id)
        if assertion is None:
            raise ValueError(f"claim {claim_id} not found on entity {entity_id}")

        from_status = assertion.status
        if not can_transition(from_status, to_status):
            return PromoteResult(
                target=target,
                from_status=from_status,
                to_status=to_status,
                op_id=ctx.op_id,
                refused=True,
                reason=explain_transition_refusal(from_status, to_status),
            )

        # Refuse promotion if verification fails.
        source_texts = {}
        for evidence in assertion.evidence:
            try:
                record = registry.load(evidence.source_id)
            except Exception:
                return PromoteResult(
                    target=target,
                    from_status=from_status,
                    to_status=to_status,
                    op_id=ctx.op_id,
                    refused=True,
                    reason=f"source {evidence.source_id} not in registry",
                )
            source_texts[evidence.source_id] = (
                workspace.root / record.raw_path
            ).read_text(encoding="utf-8")
        verification = verify_assertion(assertion, source_texts)
        if not verification.passed:
            return PromoteResult(
                target=target,
                from_status=from_status,
                to_status=to_status,
                op_id=ctx.op_id,
                refused=True,
                reason=f"verification failed: {verification.notes}",
            )

        assertion.status = to_status
        assertion.updated_at = iso_now()
        store.save_entity(entity)
        ctx.entry.touched_files.append(
            relative_posix(workspace, store.entity_path(entity_id))
        )
        return PromoteResult(
            target=target,
            from_status=from_status,
            to_status=to_status,
            op_id=ctx.op_id,
        )

