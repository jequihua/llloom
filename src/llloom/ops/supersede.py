"""`supersede` operation (Slice 078).

Direct lifecycle mutation:

```text
llloom supersede OLD --by NEW
```

Marks ``OLD`` as ``superseded`` and records a supersession link from
``NEW`` back to ``OLD`` on the canonical ``Assertion.supersedes``
field. Both targets must already be ``validated`` — the slice keeps
authority symmetric, mirroring the existing ``promote
--to superseded`` rule on the OLD side and adding a matching rule on
the NEW side so a draft / reviewed claim cannot silently retire a
validated peer.

The operation is atomic under the existing ``operation(...)`` lock /
journal contract: a refusal on the pre-mutation guard chain leaves
every entity YAML byte-identical, and the success path saves every
touched entity exactly once before the context closes.

Target grammar (shared with ``claim_card`` / ``promote`` /
``list_claims``):

- ``claim:<entity_id>:<claim_id>`` — fully qualified;
- ``<claim_id>`` — bare id, accepted only when exactly one entity
  carries that id. Ambiguous bare ids refuse with the candidate
  qualified targets listed.
"""

from __future__ import annotations

import re

from llloom.claims.lifecycle import (
    can_transition,
    explain_transition_refusal,
)
from llloom.claims.models import Assertion, EntityContainer
from llloom.claims.store import ClaimStore, ClaimStoreError
from llloom.ops._context import iso_now, operation, relative_posix
from llloom.ops.results import SupersedeResult
from llloom.workspace.layout import Workspace


class SupersedeError(Exception):
    """Raised when ``supersede(...)`` target resolution fails before
    the operation context opens (malformed target, missing or
    ambiguous bare id). The CLI catches this and prints a concise
    stderr diagnostic with exit code 1. Runtime refusal paths
    inside ``operation(...)`` surface via
    :class:`SupersedeResult.refused` / ``reason`` instead so they
    still carry an ``op_id`` for journal correlation.
    """


_QUALIFIED_TARGET_RE = re.compile(r"^claim:(?P<entity>[^:]+):(?P<claim>.+)$")
# Status the NEW claim must hold to be eligible as a replacement.
# Keeping authority symmetric: only a ``validated`` claim may
# supersede another validated claim. If the operator wants to
# supersede with a draft / reviewed claim, they should promote the
# replacement first.
_REQUIRED_NEW_STATUS = "validated"
# Target lifecycle state for the OLD claim. The shared lifecycle
# guard enforces the legal ``(validated, "superseded")`` edge so
# any other OLD status refuses at the guard with a concrete
# promotion-path hint.
_TARGET_OLD_STATUS = "superseded"


def supersede(
    workspace: Workspace,
    *,
    old: str,
    by: str,
) -> SupersedeResult:
    """Mark ``old`` as ``superseded`` and link ``by`` to it.

    See the module docstring for the lifecycle / atomicity rules.
    """
    store = ClaimStore(workspace)

    # Resolve both targets before any mutation. Resolution failures
    # raise ``SupersedeError`` and never open an ``operation(...)``
    # context (no journal entry, no lock acquisition). When both
    # targets resolve to the same entity, share a single
    # ``EntityContainer`` instance so mutations on both assertions
    # land in the same in-memory record that gets saved exactly once
    # — otherwise the second mutation would be silently dropped.
    old_entity, old_assertion = _resolve_target(store, old, label="old")
    new_entity, new_assertion = _resolve_target(
        store, by, label="--by", prefer_entity=old_entity
    )

    old_qualified = _qualified(old_entity, old_assertion)
    new_qualified = _qualified(new_entity, new_assertion)

    with operation(workspace, op_kind="supersede") as ctx:
        # Same-claim guard.
        if (
            old_entity.entity_id == new_entity.entity_id
            and old_assertion.claim_id == new_assertion.claim_id
        ):
            return _refused(
                ctx_op_id=ctx.op_id,
                old_qualified=old_qualified,
                new_qualified=new_qualified,
                old_assertion=old_assertion,
                new_assertion=new_assertion,
                reason="cannot supersede a claim with itself",
            )

        # Lifecycle guard on the OLD claim: enforce the legal
        # transition graph via the shared
        # ``llloom.claims.lifecycle`` helpers so this operation and
        # ``promote`` share a single source of truth.
        if not can_transition(old_assertion.status, _TARGET_OLD_STATUS):
            return _refused(
                ctx_op_id=ctx.op_id,
                old_qualified=old_qualified,
                new_qualified=new_qualified,
                old_assertion=old_assertion,
                new_assertion=new_assertion,
                reason=(
                    f"old claim {old_qualified}: "
                    + explain_transition_refusal(
                        old_assertion.status, _TARGET_OLD_STATUS
                    )
                ),
            )

        # Authority symmetry guard on the NEW claim: only a
        # ``validated`` replacement may supersede another claim. A
        # draft / reviewed / superseded / archived / source-cascade
        # state on the replacement side refuses.
        if new_assertion.status != _REQUIRED_NEW_STATUS:
            return _refused(
                ctx_op_id=ctx.op_id,
                old_qualified=old_qualified,
                new_qualified=new_qualified,
                old_assertion=old_assertion,
                new_assertion=new_assertion,
                reason=(
                    f"new claim {new_qualified}: status is "
                    f"{new_assertion.status!r}; replacement must be "
                    f"{_REQUIRED_NEW_STATUS!r} before it can supersede "
                    "another claim"
                ),
            )

        # Mutation phase. ``Assertion.supersedes`` carries the OLD
        # qualified target so the citation / card surface (and the
        # forthcoming Phase E doctor surface) can resolve the link
        # without a separate index.
        new_assertion.supersedes = old_qualified
        new_assertion.updated_at = iso_now()
        old_from_status = old_assertion.status
        old_assertion.status = _TARGET_OLD_STATUS
        old_assertion.updated_at = iso_now()

        # Save touched entities exactly once. When old and new live
        # on the same entity, save once.
        if old_entity.entity_id == new_entity.entity_id:
            store.save_entity(old_entity)
            ctx.entry.touched_files.append(
                relative_posix(
                    workspace, store.entity_path(old_entity.entity_id)
                )
            )
        else:
            store.save_entity(old_entity)
            store.save_entity(new_entity)
            ctx.entry.touched_files.append(
                relative_posix(
                    workspace, store.entity_path(old_entity.entity_id)
                )
            )
            ctx.entry.touched_files.append(
                relative_posix(
                    workspace, store.entity_path(new_entity.entity_id)
                )
            )

        ctx.entry.notes.append(
            f"superseded {old_qualified} -> {new_qualified}"
        )

        return SupersedeResult(
            old_target=old_qualified,
            new_target=new_qualified,
            old_from_status=old_from_status,
            old_to_status=_TARGET_OLD_STATUS,
            new_status=new_assertion.status,
            supersedes=old_qualified,
            op_id=ctx.op_id,
        )


# ---- helpers -----------------------------------------------------------


def _resolve_target(
    store: ClaimStore,
    target: str,
    *,
    label: str,
    prefer_entity: EntityContainer | None = None,
) -> tuple[EntityContainer, Assertion]:
    """Resolve ``target`` to its canonical ``(entity, assertion)``
    pair. Raises :class:`SupersedeError` on every failure mode the
    operator should see *before* an operation opens.

    ``prefer_entity`` — when supplied, a target that resolves to
    the same ``entity_id`` reuses that container instance instead
    of loading a fresh one. Lets the caller mutate both old and
    new assertions on a single in-memory entity so a single
    ``save_entity`` call persists both changes.
    """
    if not isinstance(target, str) or not target:
        raise SupersedeError(
            f"{label} target must be a non-empty string"
        )
    match = _QUALIFIED_TARGET_RE.match(target)
    if match is not None:
        entity_id = match.group("entity")
        claim_id = match.group("claim")
        if prefer_entity is not None and prefer_entity.entity_id == entity_id:
            entity = prefer_entity
        else:
            try:
                entity = store.load_entity(entity_id)
            except ClaimStoreError as exc:
                raise SupersedeError(
                    f"{label} target {target!r}: entity not found: {entity_id!r}"
                ) from exc
        assertion = entity.find_assertion(claim_id)
        if assertion is None:
            raise SupersedeError(
                f"{label} target {target!r}: claim {claim_id!r} not found "
                f"on entity {entity_id!r}"
            )
        return entity, assertion

    # Bare claim_id — walk every entity.
    candidates: list[tuple[EntityContainer, Assertion]] = []
    if prefer_entity is not None:
        assertion = prefer_entity.find_assertion(target)
        if assertion is not None:
            candidates.append((prefer_entity, assertion))
    for entity in store.iter_entities():
        if prefer_entity is not None and entity.entity_id == prefer_entity.entity_id:
            continue
        assertion = entity.find_assertion(target)
        if assertion is not None:
            candidates.append((entity, assertion))
    if not candidates:
        raise SupersedeError(
            f"{label} target {target!r}: no claim with that id found in the "
            "workspace; use the qualified form claim:<entity>:<claim> to "
            "disambiguate"
        )
    if len(candidates) > 1:
        qualified = sorted(
            _qualified(e, a) for e, a in candidates
        )
        raise SupersedeError(
            f"{label} target {target!r}: bare claim_id is ambiguous; matches "
            f"{len(candidates)} claims: {qualified}. Use the qualified form "
            "claim:<entity>:<claim> to pick one."
        )
    return candidates[0]


def _qualified(entity: EntityContainer, assertion: Assertion) -> str:
    return f"claim:{entity.entity_id}:{assertion.claim_id}"


def _refused(
    *,
    ctx_op_id: str,
    old_qualified: str,
    new_qualified: str,
    old_assertion: Assertion,
    new_assertion: Assertion,
    reason: str,
) -> SupersedeResult:
    """Build a ``SupersedeResult(refused=True, ...)`` carrying the
    journal op id so the caller can correlate the refusal with the
    no-mutation journal entry on disk.
    """
    return SupersedeResult(
        old_target=old_qualified,
        new_target=new_qualified,
        old_from_status=old_assertion.status,
        old_to_status=old_assertion.status,
        new_status=new_assertion.status,
        supersedes=old_qualified,
        op_id=ctx_op_id,
        refused=True,
        reason=reason,
    )
