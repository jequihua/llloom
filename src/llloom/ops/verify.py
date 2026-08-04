"""`verify` operation: run deterministic provenance verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llloom.claims.store import ClaimStore
from llloom.claims.verifier import verify_assertion
from llloom.ops.results import VerifyResult
from llloom.sources.registry import SourceRegistry
from llloom.workspace.layout import Workspace


def verify(
    workspace: Workspace,
    *,
    target: str | None = None,
) -> VerifyResult:
    """Verify one source, one entity, one claim, or the whole workspace.

    ``target`` forms:

    - None: verify every entity's every assertion
    - ``entity:<entity_id>``: verify all assertions on the entity
    - ``claim:<entity_id>:<claim_id>``: verify a single claim
    - ``source:<source_id>``: verify every claim citing the source
    """
    store = ClaimStore(workspace)
    registry = SourceRegistry(workspace)
    result = VerifyResult()

    source_texts = _load_all_source_texts(workspace, registry)
    to_verify = _select_targets(store, target)

    for entity_id, claim_id, assertion in to_verify:
        v = verify_assertion(assertion, source_texts)
        if v.passed:
            result.verified.append(claim_id)
        else:
            result.failed.append(claim_id)
            for note in v.notes:
                result.notes.append(f"{entity_id}.{claim_id}: {note}")
            # Surface the structured mismatch diagnostics alongside
            # the textual notes. Each mismatch carries its own claim_id
            # via the verifier; no need to re-prefix here.
            result.mismatches.extend(v.mismatches)
            result.passed = False

    return result


def _load_all_source_texts(
    workspace: Workspace, registry: SourceRegistry
) -> dict[str, str]:
    texts: dict[str, str] = {}
    for record in registry.iter_records():
        path = workspace.root / record.raw_path
        if path.is_file():
            texts[record.source_id] = path.read_text(encoding="utf-8")
    return texts


def _select_targets(
    store: ClaimStore,
    target: str | None,
) -> list[tuple[str, str, Any]]:
    out: list[tuple[str, str, Any]] = []
    if target is None:
        for entity in store.iter_entities():
            for assertion in entity.assertions:
                out.append((entity.entity_id, assertion.claim_id, assertion))
        return out

    if target.startswith("entity:"):
        eid = target.split(":", 1)[1]
        entity = store.load_entity(eid)
        for assertion in entity.assertions:
            out.append((entity.entity_id, assertion.claim_id, assertion))
        return out

    if target.startswith("claim:"):
        _, rest = target.split(":", 1)
        eid, cid = rest.split(":", 1)
        entity = store.load_entity(eid)
        assertion = entity.find_assertion(cid)
        if assertion is None:
            raise ValueError(f"claim {cid} not found on entity {eid}")
        out.append((entity.entity_id, assertion.claim_id, assertion))
        return out

    if target.startswith("source:"):
        src = target.split(":", 1)[1]
        for entity in store.iter_entities():
            for assertion in entity.assertions:
                if any(e.source_id == src for e in assertion.evidence):
                    out.append((entity.entity_id, assertion.claim_id, assertion))
        return out

    raise ValueError(
        f"unknown verify target {target!r}; expected None, entity:<id>, "
        f"claim:<entity_id>:<claim_id>, or source:<id>"
    )

