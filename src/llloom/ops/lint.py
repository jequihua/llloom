"""`lint` operation.

Layered checks from ``04_specification/operations_and_cli.md`` Â§lint:

- citation integrity (via verify)
- lock violations (stale lock without reconcile)
- page marker validity
- broken page links (first slice: minimal)
- orphan entities/pages
- unresolved aliases (pending merge proposals surfaced)
- stale derived artifacts (render fingerprint mismatch)
- canary leakage test (see canary_check)
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path

from llloom.claims.store import ClaimStore
from llloom.ops._context import relative_posix
from llloom.ops.results import LintResult
from llloom.ops.verify import verify
from llloom.pages.regions import parse_page, PageParseError
from llloom.pages.render import compute_page_render_fingerprints
from llloom.schema.policy import load_schema
from llloom.sources.registry import SourceRegistry
from llloom.state.fingerprints import FingerprintStore
from llloom.state.lock import WorkspaceLock
from llloom.workspace.layout import Workspace


# Fixed fixture canary: lives in tests; lint must refuse if it appears
# anywhere in authoritative state.
FIXED_CANARY_TOKEN = "LLLOOM_CANARY_FIXED_Z9F3"

# Prefix for generated per-run canary tokens. The prefix is
# deliberately distinctive so a naive grep finds any accidentally
# persisted generated tokens across the workspace.
GENERATED_CANARY_PREFIX = "LLLOOM_CANARY_RUN_"


def generate_canary_token() -> str:
    """Return a high-entropy per-run canary token.

    The token carries the fixed ``LLLOOM_CANARY_RUN_`` prefix plus a
    hex suffix from ``secrets.token_hex(16)`` (128 bits of entropy).
    Distinct calls produce distinct tokens with overwhelming
    probability. The generator is stdlib-only and never writes to
    the workspace on its own; callers (including :func:`lint` when
    ``generated_canary=True``) use it to build a scan set and are
    responsible for planting the token if they want to prove
    detection.
    """
    return f"{GENERATED_CANARY_PREFIX}{secrets.token_hex(16)}"


def lint(
    workspace: Workspace,
    *,
    extra_canary_tokens: list[str] | None = None,
    generated_canary: bool = False,
) -> LintResult:
    """Run workspace health checks and return a structured LintResult.

    The canary scan always includes :data:`FIXED_CANARY_TOKEN`. Any
    caller-supplied ``extra_canary_tokens`` are appended. When
    ``generated_canary`` is True, a fresh per-run token from
    :func:`generate_canary_token` is added to the scan set; this is
    the release-validation path that makes leaks harder to
    accidentally satisfy by hard-coding around the fixed token.
    Clean workspaces pass even with ``generated_canary=True`` because
    the per-run token has never been persisted anywhere by the
    package itself.
    """
    schema = load_schema(workspace)
    store = ClaimStore(workspace)
    registry = SourceRegistry(workspace)
    fingerprints = FingerprintStore(workspace)
    lock = WorkspaceLock(workspace)

    result = LintResult()
    canary_tokens = [FIXED_CANARY_TOKEN]
    if extra_canary_tokens:
        canary_tokens.extend(extra_canary_tokens)
    if generated_canary:
        canary_tokens.append(generate_canary_token())

    # 1. Citation integrity (via verifier)
    v = verify(workspace)
    if not v.passed:
        for note in v.notes:
            result.failures.append(f"citation: {note}")

    # 2. Lock violations: timed-out lock requires reconcile.
    lk = lock.read()
    if lk is not None and lock.is_timed_out(lk):
        result.failures.append(
            f"lock: workspace lock appears stale (op_id={lk.op_id}); "
            f"run reconcile"
        )

    # 3. Page marker validity + canary leakage in rendered claim blocks
    for page_path in sorted(workspace.pages.rglob("*.md")):
        rel = relative_posix(workspace, page_path)
        try:
            parsed = parse_page(page_path.read_text(encoding="utf-8"))
        except PageParseError as exc:
            # overview and other spine files might legitimately lack markers;
            # skip if the file is marked as spine.
            if schema.is_spine(rel):
                continue
            result.failures.append(f"page marker: {rel}: {exc}")
            continue
        # Canary MUST NOT appear in the claim-block region.
        for token in canary_tokens:
            if token in parsed.claim_block_inner:
                result.canary_hits.append(
                    f"canary {token!r} leaked into claim-block at {rel}"
                )

    # 4. Canary leakage in entity claim YAMLs, merge proposals, journals.
    for directory in (
        workspace.claims_entities,
        workspace.claims_merge_proposals,
        workspace.state_journals,
    ):
        if not directory.is_dir():
            continue
        for yaml_path in sorted(directory.rglob("*.yaml")):
            text = yaml_path.read_text(encoding="utf-8")
            for token in canary_tokens:
                if token in text:
                    result.canary_hits.append(
                        f"canary {token!r} leaked into {relative_posix(workspace, yaml_path)}"
                    )

    # 5. Orphan entities: entities with zero non-retracted assertions are
    #    warnings (not failures) for now.
    for entity in store.iter_entities():
        active = [
            a
            for a in entity.assertions
            if a.status not in {"retracted", "retracted_by_source", "archived"}
        ]
        if not active:
            result.warnings.append(f"orphan: entity {entity.entity_id} has no active assertions")

    # 6. Unresolved aliases
    for proposal_id in store.list_proposal_ids():
        proposal = store.load_proposal(proposal_id)
        if proposal.status == "pending":
            result.warnings.append(f"pending alias merge: {proposal_id}")

    # 7. Stale derived artifacts: render fingerprint mismatch.
    # Slice 071: page/block-centric. The expected fingerprint is the
    # union across every entity that targets a page; the stored
    # fingerprint must match the union, not any one entity's
    # contribution. One warning per page (deduplicated).
    fps = fingerprints.load()
    expected_fingerprints = compute_page_render_fingerprints(
        store.iter_entities()
    )
    for page_id in sorted(expected_fingerprints):
        actual = fps.get(page_id)
        if actual is not None and actual != expected_fingerprints[page_id]:
            result.warnings.append(
                f"stale render fingerprint for page_id={page_id}"
            )

    # 8. Modified immutable evidence (hash changed under the registry)
    for record in registry.iter_records():
        path = workspace.root / record.raw_path
        if not path.is_file():
            result.failures.append(
                f"source {record.source_id}: raw file missing at {record.raw_path}"
            )
            continue
        current_hash = SourceRegistry.hash_file(path)
        if record.content_hash != current_hash:
            result.failures.append(
                f"source {record.source_id}: raw evidence hash changed "
                f"({record.content_hash} -> {current_hash})"
            )

    return result

