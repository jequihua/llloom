"""Alias merge queue verbs.

Write-as-new, queue-for-merge: the ingest path writes a new entity and
files a ``MergeProposal``. Nothing is merged silently. These verbs drive
the review workflow:

- ``list_merge_proposals``: list pending proposals
- ``review_alias``: approve or reject a proposal
- ``merge_alias``: apply an approved proposal
- ``reject_alias``: close a proposal as rejected
"""

from __future__ import annotations

from llloom.claims.models import Alias
from llloom.claims.store import ClaimStore
from llloom.ops._context import iso_now, operation, relative_posix
from llloom.ops.results import MergeProposalSummary
from llloom.workspace.layout import Workspace


def list_merge_proposals(workspace: Workspace) -> list[MergeProposalSummary]:
    store = ClaimStore(workspace)
    out: list[MergeProposalSummary] = []
    for pid in store.list_proposal_ids():
        p = store.load_proposal(pid)
        out.append(
            MergeProposalSummary(
                proposal_id=p.proposal_id,
                source_entity_id=p.source_entity_id,
                target_entity_id=p.target_entity_id,
                status=p.status,
                proposed_alias_text=p.proposed_alias_text,
            )
        )
    return out


def review_alias(
    workspace: Workspace,
    *,
    proposal_id: str,
    decision: str,
    notes: str | None = None,
) -> MergeProposalSummary:
    if decision not in {"approve", "reject"}:
        raise ValueError(f"decision must be approve|reject, got {decision!r}")
    store = ClaimStore(workspace)
    with operation(workspace, op_kind=f"review_alias.{decision}") as ctx:
        proposal = store.load_proposal(proposal_id)
        if proposal.status != "pending":
            raise ValueError(
                f"proposal {proposal_id} not pending (status={proposal.status})"
            )
        proposal.status = "approved" if decision == "approve" else "rejected"
        proposal.reviewed_at = iso_now()
        proposal.review_notes = notes
        store.save_proposal(proposal)
        ctx.entry.touched_files.append(
            relative_posix(workspace, store.proposal_path(proposal_id))
        )
        return MergeProposalSummary(
            proposal_id=proposal.proposal_id,
            source_entity_id=proposal.source_entity_id,
            target_entity_id=proposal.target_entity_id,
            status=proposal.status,
            proposed_alias_text=proposal.proposed_alias_text,
        )


def merge_alias(
    workspace: Workspace,
    *,
    proposal_id: str,
) -> MergeProposalSummary:
    """Apply an approved alias merge.

    Rules:

    - refuses unless the proposal was approved via ``review_alias``
    - never rewrites raw evidence
    - adds the proposed alias text to the target entity's ``aliases``
    - marks the source entity ``merged_into``
    - leaves the source's assertions in place (callers may later
      consolidate them; the first slice keeps the conservative behavior)
    """
    store = ClaimStore(workspace)
    with operation(workspace, op_kind="merge_alias") as ctx:
        proposal = store.load_proposal(proposal_id)
        if proposal.status != "approved":
            raise ValueError(
                f"proposal {proposal_id} not approved (status={proposal.status})"
            )
        target = store.load_entity(proposal.target_entity_id)
        existing_aliases = {a.alias_text.strip().lower() for a in target.aliases}
        if proposal.proposed_alias_text.strip().lower() not in existing_aliases:
            target.aliases.append(
                Alias(
                    alias_id=f"a.{proposal.proposal_id}",
                    alias_text=proposal.proposed_alias_text,
                    status="active",
                )
            )
            store.save_entity(target)
            ctx.entry.touched_files.append(
                relative_posix(workspace, store.entity_path(target.entity_id))
            )
        if store.exists(proposal.source_entity_id):
            source_entity = store.load_entity(proposal.source_entity_id)
            source_entity.status = "merged_into"
            store.save_entity(source_entity)
            ctx.entry.touched_files.append(
                relative_posix(workspace, store.entity_path(source_entity.entity_id))
            )
        proposal.status = "applied"
        proposal.reviewed_at = iso_now()
        store.save_proposal(proposal)
        ctx.entry.touched_files.append(
            relative_posix(workspace, store.proposal_path(proposal_id))
        )
        return MergeProposalSummary(
            proposal_id=proposal.proposal_id,
            source_entity_id=proposal.source_entity_id,
            target_entity_id=proposal.target_entity_id,
            status=proposal.status,
            proposed_alias_text=proposal.proposed_alias_text,
        )


def reject_alias(
    workspace: Workspace,
    *,
    proposal_id: str,
    notes: str | None = None,
) -> MergeProposalSummary:
    return review_alias(
        workspace, proposal_id=proposal_id, decision="reject", notes=notes
    )

