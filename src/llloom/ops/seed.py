"""`seed apply` operation: deterministic seed manifest application.

Slice 075 added the first-class seed-manifest entry point. A
manifest is a YAML document declaring sources and their candidate
claims; the apply function registers each source, validates every
claim through the existing verifier path, and persists the
verified batch via the same atomic claim-store primitives that
:func:`llloom.ops.ingest.ingest` uses for deterministic
``seed_claims`` — **never** invoking a model, ``LLMInvoke``, or
``NullModel``.

The user-facing contract:

- ``llloom seed apply <manifest.yaml>`` (or
  :func:`apply_seed_manifest`) registers sources and persists
  every verified claim; the post-apply render step matches the
  ingest path's behavior;
- ``--dry-run`` validates and previews the work but writes nothing
  (no source registry update, no claim YAML, no page render, no
  fingerprint write, no journal entry — including no journal entry
  for the seed apply itself);
- ``--no-render`` persists verified claims but suppresses page
  rendering;
- ``--status <status>`` supplies an operation-level default
  lifecycle status that fills in for claims without an explicit
  merged status;
- the journal records ``op_kind="seed_apply"``, notes name the
  deterministic-no-provider mode, and ``invocation_logs`` stays
  empty.

See ``04_specification/seed_manifest_v1.md`` for the schema and
merge contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from llloom.claims.locators import SpanResolutionError, resolve_span
from llloom.claims.models import CLAIM_STATUSES, Locator
from llloom.claims.store import ClaimStore
from llloom.ops._context import operation, relative_posix
from llloom.ops.ingest import (
    ClaimVerificationError,
    SeedClaim,
    _apply_candidates,
    _render_targets,
)
from llloom.ops.results import (
    CreatedClaim,
    PlannedSeedClaim,
    SeedManifestResult,
)
from llloom.schema.policy import load_schema
from llloom.sources.registry import SourceRegistry, SourceRegistryError
from llloom.state.journal import OperationJournal
from llloom.state.seed_reports import (
    EXCERPT_EQUALITY_MODES,
    SEED_UPDATE_REPORT_VERSION,
    SeedExcerptCheck,
    check_seed_excerpt_equality,
    excerpt_check_to_mapping,
    hash_file_sha256,
    iso_now,
    write_seed_update_report,
)
from llloom.workspace.layout import Workspace


_SEED_MANIFEST_VERSION = "seed_manifest_v1"
_OP_KIND_SEED_APPLY = "seed_apply"
_REQUIRED_LOCATOR_FIELDS: tuple[str, ...] = (
    "locator_type",
    "heading_path",
    "paragraph_index",
    "sentence_start",
    "sentence_end",
)
# Slice 076: optional per-claim excerpt-equality field. Allowed
# values flow through ``_validate_excerpt_equality_if_present`` at
# every merge level (manifest defaults → source defaults → claim).
# The default is ``"none"``; ``"exact_one_sentence"`` opts the claim
# into the deterministic one-sentence equality check.
_EXCERPT_EQUALITY_FIELD = "excerpt_equality"


class SeedManifestError(Exception):
    """Raised when a seed manifest cannot be parsed, merged, or
    validated. Surfaced via :class:`SeedManifestResult.refusal_reason`
    as a refused result — never propagates out of
    :func:`apply_seed_manifest`.
    """


def apply_seed_manifest(
    workspace: Workspace,
    manifest_path: Path | str,
    *,
    dry_run: bool = False,
    no_render: bool = False,
    status: str | None = None,
) -> SeedManifestResult:
    """Deterministic seed-manifest application.

    The function never invokes a model. Dry-run writes nothing
    (no source registry update, no claim YAML, no page, no
    fingerprint, no lock, no journal entry). The real-apply path
    registers each source, calls ``_apply_candidates(...)`` per
    source for batch-atomic claim persistence, and renders via
    the same ``_render_targets(...)`` helper the ingest path uses
    unless ``no_render=True``.
    """
    manifest_path_abs = Path(manifest_path).resolve()
    try:
        manifest_relpath = manifest_path_abs.relative_to(workspace.root.resolve()).as_posix()
    except ValueError:
        manifest_relpath = str(manifest_path_abs)

    if status is not None and status not in CLAIM_STATUSES:
        return _refused(
            manifest_path=manifest_relpath,
            dry_run=dry_run,
            no_render=no_render,
            refusal_reason=(
                f"invalid --status {status!r}; allowed lifecycle "
                f"states: {sorted(CLAIM_STATUSES)}"
            ),
        )

    if not manifest_path_abs.is_file():
        return _refused(
            manifest_path=manifest_relpath,
            dry_run=dry_run,
            no_render=no_render,
            refusal_reason=f"manifest file not found: {manifest_path_abs}",
        )

    try:
        raw = yaml.safe_load(manifest_path_abs.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return _refused(
            manifest_path=manifest_relpath,
            dry_run=dry_run,
            no_render=no_render,
            refusal_reason=f"manifest YAML parse error: {exc}",
        )

    try:
        parsed_sources = _parse_and_merge(raw, cli_status=status)
    except SeedManifestError as exc:
        return _refused(
            manifest_path=manifest_relpath,
            dry_run=dry_run,
            no_render=no_render,
            refusal_reason=str(exc),
        )

    # Preflight every source's path and readability before any write.
    for source in parsed_sources:
        rel_path = source["path"]
        if Path(rel_path).is_absolute():
            return _refused(
                manifest_path=manifest_relpath,
                dry_run=dry_run,
                no_render=no_render,
                refusal_reason=(
                    f"source path {rel_path!r} is absolute; manifest "
                    "source paths must be workspace-relative"
                ),
            )
        resolved = (workspace.root / rel_path).resolve()
        try:
            resolved.relative_to(workspace.root.resolve())
        except ValueError:
            return _refused(
                manifest_path=manifest_relpath,
                dry_run=dry_run,
                no_render=no_render,
                refusal_reason=(
                    f"source path {rel_path!r} resolves outside the "
                    f"workspace ({resolved})"
                ),
            )
        if not resolved.is_file():
            return _refused(
                manifest_path=manifest_relpath,
                dry_run=dry_run,
                no_render=no_render,
                refusal_reason=f"source file not found: {rel_path}",
            )
        source["_resolved_path"] = resolved

    # Slice 075a: preflight every source's id, source_class, and
    # existing-registry conflict **before** any persistence (and
    # before any ``operation(...)`` context opens). This converts
    # predictable registration problems — invalid id syntax, unknown
    # source class, hash mismatch with an already-registered source,
    # class mismatch with an already-registered source — into
    # structured ``SeedManifestResult.refusal_reason`` results
    # instead of letting ``SourceRegistryError`` propagate out of
    # ``operation(...)`` (which would correctly leave the journal
    # ``in_progress`` and the lock held, but the Slice 075 architect
    # review flagged that as wrong for manifest-shape problems).
    try:
        _preflight_source_registration(workspace, parsed_sources)
    except SeedManifestError as exc:
        return _refused(
            manifest_path=manifest_relpath,
            dry_run=dry_run,
            no_render=no_render,
            refusal_reason=str(exc),
        )

    # Build PlannedSeedClaim records up front so the result reports
    # them even when a later source's Phase 1 verifier refuses.
    sources_planned: list[str] = [s["source_id"] for s in parsed_sources]
    claims_planned: list[PlannedSeedClaim] = []
    for source in parsed_sources:
        for claim in source["claims"]:
            claims_planned.append(
                PlannedSeedClaim(
                    source_id=source["source_id"],
                    claim_id=claim["claim_id"],
                    entity_id=claim["entity_id"],
                    status=claim["status"],
                )
            )

    if dry_run:
        return _dry_run_validate(
            workspace=workspace,
            manifest_relpath=manifest_relpath,
            parsed_sources=parsed_sources,
            sources_planned=sources_planned,
            claims_planned=claims_planned,
            no_render=no_render,
        )

    return _apply(
        workspace=workspace,
        manifest_relpath=manifest_relpath,
        parsed_sources=parsed_sources,
        sources_planned=sources_planned,
        claims_planned=claims_planned,
        no_render=no_render,
    )


# ---- parsing + merging -------------------------------------------------


def _parse_and_merge(raw: Any, *, cli_status: str | None) -> list[dict[str, Any]]:
    """Validate the manifest top-level shape, merge defaults across the
    three levels, and return a list of source records ready for the
    apply / dry-run paths.

    Each returned source dict carries:

    - ``source_id``, ``path``, ``source_class``
    - ``claims`` — list of merged claim dicts, each with ``entity_id``,
      ``entity_type``, ``display_name``, ``claim_id``, ``claim_kind``,
      ``claim_text``, ``locator`` (Locator instance),
      ``render_target`` ((page_id, block_id) tuple), and the effective
      ``status``.
    """
    if not isinstance(raw, dict):
        raise SeedManifestError("manifest root must be a YAML mapping")
    version = raw.get("version")
    if version != _SEED_MANIFEST_VERSION:
        raise SeedManifestError(
            f"unsupported manifest version {version!r}; "
            f"expected {_SEED_MANIFEST_VERSION!r}"
        )
    manifest_defaults = raw.get("defaults") or {}
    if not isinstance(manifest_defaults, dict):
        raise SeedManifestError("manifest 'defaults' must be a mapping")
    _validate_status_if_present(manifest_defaults, where="manifest defaults")
    _validate_excerpt_equality_if_present(
        manifest_defaults, where="manifest defaults"
    )

    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise SeedManifestError(
            "manifest 'sources' must be a non-empty list of source entries"
        )

    parsed: list[dict[str, Any]] = []
    for index, source_raw in enumerate(sources_raw):
        if not isinstance(source_raw, dict):
            raise SeedManifestError(
                f"sources[{index}] must be a mapping"
            )
        source_defaults = source_raw.get("defaults") or {}
        if not isinstance(source_defaults, dict):
            raise SeedManifestError(
                f"sources[{index}].defaults must be a mapping"
            )
        _validate_status_if_present(
            source_defaults, where=f"sources[{index}].defaults"
        )
        _validate_excerpt_equality_if_present(
            source_defaults, where=f"sources[{index}].defaults"
        )

        merged_source = _merge_shallow(manifest_defaults, source_defaults)
        # Source-level overrides at the source record itself.
        for field in ("path", "source_id", "source_class"):
            if field in source_raw:
                merged_source[field] = source_raw[field]
        for required in ("path", "source_id", "source_class"):
            if not isinstance(merged_source.get(required), str) or not merged_source[required]:
                raise SeedManifestError(
                    f"sources[{index}] missing required field {required!r} "
                    "after merging manifest + source defaults"
                )

        claims_raw = source_raw.get("claims")
        if not isinstance(claims_raw, list) or not claims_raw:
            raise SeedManifestError(
                f"sources[{index}].claims must be a non-empty list"
            )

        merged_claims: list[dict[str, Any]] = []
        for claim_index, claim_raw in enumerate(claims_raw):
            if not isinstance(claim_raw, dict):
                raise SeedManifestError(
                    f"sources[{index}].claims[{claim_index}] must be a mapping"
                )
            _validate_status_if_present(
                claim_raw,
                where=f"sources[{index}].claims[{claim_index}]",
            )
            _validate_excerpt_equality_if_present(
                claim_raw,
                where=f"sources[{index}].claims[{claim_index}]",
            )
            merged = _merge_claim(
                manifest_defaults=manifest_defaults,
                source_defaults=source_defaults,
                claim=claim_raw,
                cli_status=cli_status,
            )
            for required in (
                "entity_id",
                "entity_type",
                "display_name",
                "claim_id",
                "claim_kind",
                "claim_text",
                "status",
                "locator",
                "render_target",
            ):
                if required not in merged or merged[required] in (None, ""):
                    raise SeedManifestError(
                        f"sources[{index}].claims[{claim_index}] missing "
                        f"required field {required!r} after merging "
                        "manifest + source + claim defaults"
                    )
            # Coerce locator / render_target to typed shapes used by
            # the existing _apply_candidates pipeline.
            merged["locator"] = _coerce_locator(
                merged["locator"],
                where=f"sources[{index}].claims[{claim_index}].locator",
            )
            merged["render_target"] = _coerce_render_target(
                merged["render_target"],
                where=f"sources[{index}].claims[{claim_index}].render_target",
            )
            # Slice 076: resolve the effective excerpt_equality mode.
            # Default to "none"; per-claim override wins over source
            # default which wins over manifest default (the shallow
            # merge already produced the right value if any level set
            # the field). Validate the effective value against the
            # known modes so a typo refuses before any persistence.
            effective_mode = merged.get(_EXCERPT_EQUALITY_FIELD) or "none"
            if effective_mode not in EXCERPT_EQUALITY_MODES:
                raise SeedManifestError(
                    f"invalid effective excerpt_equality {effective_mode!r} "
                    f"for claim {merged.get('claim_id', '?')!r}; allowed "
                    f"modes: {sorted(EXCERPT_EQUALITY_MODES)}"
                )
            merged[_EXCERPT_EQUALITY_FIELD] = effective_mode
            merged_claims.append(merged)

        parsed.append(
            {
                "source_id": merged_source["source_id"],
                "path": merged_source["path"],
                "source_class": merged_source["source_class"],
                "claims": merged_claims,
            }
        )

    return parsed


def _merge_shallow(*levels: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge: later levels override earlier ones. ``locator``
    and ``render_target`` get a nested mapping merge so callers can
    declare partial overrides without restating every field.
    """
    out: dict[str, Any] = {}
    for level in levels:
        for key, value in level.items():
            if key in ("locator", "render_target") and isinstance(value, dict):
                base = out.get(key) or {}
                if not isinstance(base, dict):
                    base = {}
                out[key] = {**base, **value}
            else:
                out[key] = value
    return out


def _merge_claim(
    *,
    manifest_defaults: dict[str, Any],
    source_defaults: dict[str, Any],
    claim: dict[str, Any],
    cli_status: str | None,
) -> dict[str, Any]:
    """Apply the four-level merge: manifest → source → claim → CLI status."""
    merged = _merge_shallow(manifest_defaults, source_defaults, claim)
    if "status" not in merged or merged["status"] in (None, ""):
        if cli_status is not None:
            merged["status"] = cli_status
        else:
            merged["status"] = "draft"
    if merged["status"] not in CLAIM_STATUSES:
        raise SeedManifestError(
            f"invalid effective status {merged['status']!r} for claim "
            f"{merged.get('claim_id', '?')!r}; allowed lifecycle "
            f"states: {sorted(CLAIM_STATUSES)}"
        )
    return merged


def _validate_status_if_present(
    record: dict[str, Any], *, where: str
) -> None:
    if "status" in record and record["status"] is not None:
        status = record["status"]
        if not isinstance(status, str) or status not in CLAIM_STATUSES:
            raise SeedManifestError(
                f"invalid status {status!r} in {where}; allowed "
                f"lifecycle states: {sorted(CLAIM_STATUSES)}"
            )


def _validate_excerpt_equality_if_present(
    record: dict[str, Any], *, where: str
) -> None:
    """Slice 076: refuse a manifest that declares an unknown
    ``excerpt_equality`` mode at any merge level. Absent / None is
    allowed and defaults to ``"none"``.
    """
    if _EXCERPT_EQUALITY_FIELD not in record:
        return
    mode = record[_EXCERPT_EQUALITY_FIELD]
    if mode is None:
        return
    if not isinstance(mode, str) or mode not in EXCERPT_EQUALITY_MODES:
        raise SeedManifestError(
            f"invalid excerpt_equality {mode!r} in {where}; allowed "
            f"modes: {sorted(EXCERPT_EQUALITY_MODES)}"
        )


def _coerce_locator(raw: Any, *, where: str) -> Locator:
    if isinstance(raw, Locator):
        return raw
    if not isinstance(raw, dict):
        raise SeedManifestError(f"{where} must be a mapping")
    for field in _REQUIRED_LOCATOR_FIELDS:
        if field not in raw:
            raise SeedManifestError(
                f"{where} missing required locator field {field!r}"
            )
    return Locator(
        locator_type=str(raw["locator_type"]),
        heading_path=list(raw.get("heading_path") or []),
        paragraph_index=int(raw["paragraph_index"]),
        sentence_start=int(raw["sentence_start"]),
        sentence_end=int(raw["sentence_end"]),
        # The existing Locator dataclass tolerates extra optional
        # fields via its dataclass defaults; pass through any
        # code_v1 fields if a future manifest declares them.
        path=raw.get("path"),
        start_line=raw.get("start_line"),
        start_col=raw.get("start_col"),
        end_line=raw.get("end_line"),
        end_col=raw.get("end_col"),
    )


def _coerce_render_target(raw: Any, *, where: str) -> tuple[str, str]:
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        return (str(raw[0]), str(raw[1]))
    if not isinstance(raw, dict):
        raise SeedManifestError(f"{where} must be a mapping or 2-element list")
    page_id = raw.get("page_id")
    block_id = raw.get("block_id")
    if not isinstance(page_id, str) or not isinstance(block_id, str):
        raise SeedManifestError(
            f"{where} must have string 'page_id' and 'block_id' fields"
        )
    return (page_id, block_id)


# ---- dry-run + apply ---------------------------------------------------


def _dry_run_validate(
    *,
    workspace: Workspace,
    manifest_relpath: str,
    parsed_sources: list[dict[str, Any]],
    sources_planned: list[str],
    claims_planned: list[PlannedSeedClaim],
    no_render: bool,
) -> SeedManifestResult:
    """Verifier-grade dry-run with zero workspace mutation.

    Reads each source's raw bytes directly (no registry write) and
    calls ``_apply_candidates(..., dry_run=True)`` per source so the
    full Phase 1 verifier walk runs against every claim. Refuses on
    any verifier failure with a descriptive ``refusal_reason``.

    Slice 076: also runs :func:`check_seed_excerpt_equality` per
    claim. A mismatch refuses the source batch (matching real-apply
    behavior) so a manifest with a broken excerpt-equality assertion
    fails fast on dry-run too. The check is pure; no report file is
    written on the dry-run path.
    """
    store = ClaimStore(workspace)
    excerpt_checks: list[SeedExcerptCheck] = []
    for source in parsed_sources:
        raw_path: Path = source["_resolved_path"]
        try:
            raw_text = raw_path.read_text(encoding="utf-8")
        except OSError as exc:
            return _refused(
                manifest_path=manifest_relpath,
                dry_run=True,
                no_render=no_render,
                sources_planned=sources_planned,
                claims_planned=claims_planned,
                refusal_reason=(
                    f"source {source['source_id']!r} read error: {exc}"
                ),
            )
        seed_claims = [_to_seed_claim(c) for c in source["claims"]]
        try:
            _apply_candidates(
                workspace=workspace,
                store=store,
                source_id=source["source_id"],
                candidates=seed_claims,
                source_text=raw_text,
                seed_claim_status="draft",  # already merged into each candidate
                dry_run=True,
            )
        except ClaimVerificationError as exc:
            return _refused(
                manifest_path=manifest_relpath,
                dry_run=True,
                no_render=no_render,
                sources_planned=sources_planned,
                claims_planned=claims_planned,
                refusal_reason=(
                    f"source {source['source_id']!r} verifier refused: {exc}"
                ),
            )

        # Slice 076: dry-run runs the deterministic excerpt-equality
        # check too so a mismatching manifest fails fast and never
        # persists. The check is pure (no I/O); we record the result
        # in ``excerpt_checks`` so dry-run callers can inspect the
        # planned previews before committing.
        for claim in source["claims"]:
            mode = claim.get(_EXCERPT_EQUALITY_FIELD, "none")
            try:
                resolved_excerpt = resolve_span(claim["locator"], raw_text)
            except SpanResolutionError as exc:
                return _refused(
                    manifest_path=manifest_relpath,
                    dry_run=True,
                    no_render=no_render,
                    sources_planned=sources_planned,
                    claims_planned=claims_planned,
                    refusal_reason=(
                        f"source {source['source_id']!r} locator did not "
                        f"resolve for claim {claim['claim_id']!r}: {exc}"
                    ),
                )
            check = check_seed_excerpt_equality(
                claim_id=claim["claim_id"],
                claim_text=claim["claim_text"],
                resolved_excerpt=resolved_excerpt,
                mode=mode,
            )
            excerpt_checks.append(check)
            if not check.matched:
                return _refused(
                    manifest_path=manifest_relpath,
                    dry_run=True,
                    no_render=no_render,
                    sources_planned=sources_planned,
                    claims_planned=claims_planned,
                    refusal_reason=(
                        f"source {source['source_id']!r} excerpt_equality "
                        f"refused: {check.message}"
                    ),
                )

    return SeedManifestResult(
        manifest_path=manifest_relpath,
        dry_run=True,
        no_render=no_render,
        sources_planned=sources_planned,
        claims_planned=claims_planned,
        excerpt_checks=excerpt_checks,
    )


def _apply(
    *,
    workspace: Workspace,
    manifest_relpath: str,
    parsed_sources: list[dict[str, Any]],
    sources_planned: list[str],
    claims_planned: list[PlannedSeedClaim],
    no_render: bool,
) -> SeedManifestResult:
    """Real apply path: register each source, persist verified claims,
    optionally render.

    Each source runs inside its own ``operation(...)`` context so the
    journal carries one entry per source. The V1 atomicity boundary
    is per-source-entry: a verifier failure in source N refuses that
    source's batch but does not roll back sources 1..N-1 that
    persisted successfully.

    Per-source preflight: every source's claims are validated via
    ``_apply_candidates(..., dry_run=True)`` **before** the source's
    ``operation(...)`` context is entered. This converts a verifier
    failure into a structured ``SeedManifestResult.refusal_reason``
    instead of letting :class:`ClaimVerificationError` propagate out
    of ``operation(...)`` (which would leave the journal
    ``in_progress`` and the lock held — correct for genuine write
    interruptions, but wrong for a manifest-shape problem the agent
    needs to fix).
    """
    op_ids: list[str] = []
    claims_created: list[CreatedClaim] = []
    entities_touched: list[str] = []
    pages_rendered: list[str] = []
    excerpt_checks: list[SeedExcerptCheck] = []
    store = ClaimStore(workspace)

    # Slice 076: snapshot before-counts so the report can carry a
    # before/after delta. Captured once outside any operation context
    # so the snapshot reflects the workspace state the caller saw.
    counts_before = _snapshot_workspace_counts(workspace)

    # Per-source verifier preflight before any persistence. Slice 076
    # additionally runs the deterministic excerpt-equality check for
    # any claim with ``excerpt_equality: exact_one_sentence``. A
    # mismatch refuses the source batch atomically with a bounded
    # diagnostic; no journal entry, no source/claim/page mutation.
    for source in parsed_sources:
        raw_path: Path = source["_resolved_path"]
        try:
            raw_text = raw_path.read_text(encoding="utf-8")
        except OSError as exc:
            return _refused(
                manifest_path=manifest_relpath,
                dry_run=False,
                no_render=no_render,
                sources_planned=sources_planned,
                claims_planned=claims_planned,
                refusal_reason=(
                    f"source {source['source_id']!r} read error: {exc}"
                ),
            )
        try:
            _apply_candidates(
                workspace=workspace,
                store=store,
                source_id=source["source_id"],
                candidates=[_to_seed_claim(c) for c in source["claims"]],
                source_text=raw_text,
                seed_claim_status="draft",
                dry_run=True,
            )
        except ClaimVerificationError as exc:
            return _refused(
                manifest_path=manifest_relpath,
                dry_run=False,
                no_render=no_render,
                sources_planned=sources_planned,
                claims_planned=claims_planned,
                refusal_reason=(
                    f"source {source['source_id']!r} verifier refused: {exc}"
                ),
            )

        # Slice 076: deterministic excerpt-equality preflight per
        # source. Resolve each claim's locator against the same raw
        # source text the verifier just walked. The helper short-
        # circuits modes other than ``exact_one_sentence`` so the
        # default (``none``) path is byte-equivalent to Slice 075.
        for claim in source["claims"]:
            mode = claim.get(_EXCERPT_EQUALITY_FIELD, "none")
            try:
                resolved_excerpt = resolve_span(claim["locator"], raw_text)
            except SpanResolutionError as exc:
                # The verifier preflight above would already have
                # caught this; the guard is here so this preflight
                # never raises a bare SpanResolutionError.
                return _refused(
                    manifest_path=manifest_relpath,
                    dry_run=False,
                    no_render=no_render,
                    sources_planned=sources_planned,
                    claims_planned=claims_planned,
                    refusal_reason=(
                        f"source {source['source_id']!r} locator did not "
                        f"resolve for claim {claim['claim_id']!r}: {exc}"
                    ),
                )
            check = check_seed_excerpt_equality(
                claim_id=claim["claim_id"],
                claim_text=claim["claim_text"],
                resolved_excerpt=resolved_excerpt,
                mode=mode,
            )
            excerpt_checks.append(check)
            if not check.matched:
                return _refused(
                    manifest_path=manifest_relpath,
                    dry_run=False,
                    no_render=no_render,
                    sources_planned=sources_planned,
                    claims_planned=claims_planned,
                    refusal_reason=(
                        f"source {source['source_id']!r} excerpt_equality "
                        f"refused: {check.message}"
                    ),
                )

    for source in parsed_sources:
        raw_path = source["_resolved_path"]
        seed_claims = [_to_seed_claim(c) for c in source["claims"]]
        registry = SourceRegistry(workspace)

        with operation(
            workspace,
            op_kind=_OP_KIND_SEED_APPLY,
            owner_id=f"seed_apply.{source['source_id']}",
        ) as ctx:
            ctx.entry.notes.append(
                f"deterministic seed manifest: {manifest_relpath}"
            )
            ctx.entry.notes.append("provider: none")
            ctx.entry.notes.append(
                f"source_id={source['source_id']!r} "
                f"source_class={source['source_class']!r}"
            )
            op_ids.append(ctx.op_id)

            # Slice 075a: predictable registration errors (invalid id,
            # unknown source_class, hash/class conflict with existing
            # record) are filtered by ``_preflight_source_registration``
            # before this context opens. Any ``SourceRegistryError``
            # raised here would be a genuine race / on-disk corruption
            # case and is allowed to propagate — the existing
            # ``operation(...)`` contract leaves the journal
            # ``in_progress`` and the lock held so ``reconcile`` can
            # triage it as a real write interruption.
            record, _reg_state = registry.register(
                source_id=source["source_id"],
                raw_path=raw_path,
                source_class=source["source_class"],
            )
            ctx.entry.touched_files.append(
                relative_posix(workspace, registry.record_path(source["source_id"]))
            )

            raw_text = raw_path.read_text(encoding="utf-8")
            created, entities, _proposals = _apply_candidates(
                workspace=workspace,
                store=store,
                source_id=source["source_id"],
                candidates=seed_claims,
                source_text=raw_text,
                seed_claim_status="draft",
            )
            claims_created.extend(created)
            for eid in entities:
                if eid not in entities_touched:
                    entities_touched.append(eid)
            ctx.entry.touched_files.extend(
                f"claims/entities/{eid}.yaml" for eid in entities
            )

            if not no_render and entities:
                pages = _render_targets(workspace, store, entities)
                pages_rendered.extend(pages)
                ctx.entry.touched_files.extend(pages)
            elif no_render:
                ctx.entry.notes.append(
                    "render skipped at caller request (no_render=True)"
                )

    # Slice 076: snapshot after-counts and write the durable update
    # report. The report file is keyed by the first source's op_id so
    # a multi-source apply still maps to one report — the journal
    # already retains the per-source op_ids list.
    counts_after = _snapshot_workspace_counts(workspace)
    report_relpath: str | None = None
    if op_ids:
        report_op_id = op_ids[0]
        report_payload = _build_update_report(
            workspace=workspace,
            manifest_relpath=manifest_relpath,
            op_ids=op_ids,
            parsed_sources=parsed_sources,
            sources_planned=sources_planned,
            claims_planned=claims_planned,
            claims_created=claims_created,
            entities_touched=sorted(set(entities_touched)),
            pages_rendered=pages_rendered,
            no_render=no_render,
            excerpt_checks=excerpt_checks,
            counts_before=counts_before,
            counts_after=counts_after,
        )
        target = write_seed_update_report(
            workspace, op_id=report_op_id, payload=report_payload
        )
        report_relpath = relative_posix(workspace, target)

    return SeedManifestResult(
        manifest_path=manifest_relpath,
        dry_run=False,
        no_render=no_render,
        sources_planned=sources_planned,
        claims_planned=claims_planned,
        claims_created=claims_created,
        entities_touched=sorted(set(entities_touched)),
        pages_rendered=pages_rendered,
        render_skipped=no_render,
        op_ids=op_ids,
        report_path=report_relpath,
        excerpt_checks=excerpt_checks,
        counts_before=counts_before,
        counts_after=counts_after,
    )


def _to_seed_claim(merged_claim: dict[str, Any]) -> SeedClaim:
    return SeedClaim(
        entity_id=merged_claim["entity_id"],
        entity_type=merged_claim["entity_type"],
        display_name=merged_claim["display_name"],
        claim_id=merged_claim["claim_id"],
        claim_kind=merged_claim["claim_kind"],
        claim_text=merged_claim["claim_text"],
        locator=merged_claim["locator"],
        render_target=merged_claim["render_target"],
        status=merged_claim["status"],
    )


def _refused(
    *,
    manifest_path: str,
    dry_run: bool,
    no_render: bool,
    refusal_reason: str,
    sources_planned: list[str] | None = None,
    claims_planned: list[PlannedSeedClaim] | None = None,
) -> SeedManifestResult:
    return SeedManifestResult(
        manifest_path=manifest_path,
        dry_run=dry_run,
        no_render=no_render,
        sources_planned=list(sources_planned or []),
        claims_planned=list(claims_planned or []),
        refusal_reason=refusal_reason,
    )


def _preflight_source_registration(
    workspace: Workspace, parsed_sources: list[dict[str, Any]]
) -> None:
    """Slice 075a: validate every source's id syntax, source_class
    membership in the schema, and existing-registry conflicts before
    any ``operation(...)`` context opens.

    Raises :class:`SeedManifestError` (which the caller converts to
    ``SeedManifestResult.refusal_reason``). No mutation, no
    journal entry, no lock acquisition.
    """
    schema = load_schema(workspace)
    known_classes = set(schema.source_classes)
    registry = SourceRegistry(workspace)
    seen_ids: set[str] = set()

    for source in parsed_sources:
        source_id = source["source_id"]
        source_class = source["source_class"]
        raw_path: Path = source["_resolved_path"]

        if source_id in seen_ids:
            raise SeedManifestError(
                f"manifest declares source_id {source_id!r} more than once; "
                "each source entry must use a unique id"
            )
        seen_ids.add(source_id)

        # Id syntax. SourceRegistry.validate_source_id raises
        # SourceRegistryError; convert to SeedManifestError so the
        # caller routes it through the structured refusal path.
        try:
            SourceRegistry.validate_source_id(source_id)
        except SourceRegistryError as exc:
            raise SeedManifestError(
                f"source {source_id!r}: invalid source_id: {exc}"
            ) from exc

        # Source class membership in the workspace schema.
        if source_class not in known_classes:
            raise SeedManifestError(
                f"source {source_id!r}: unknown source_class "
                f"{source_class!r}; known classes: {sorted(known_classes)}"
            )

        # Existing-registry conflicts. The registry's authoritative
        # register(...) refuses these inside operation(...); preflight
        # the same checks here so the refusal lands as a structured
        # SeedManifestResult instead of stranding the lock.
        if registry.exists(source_id):
            try:
                existing = registry.load(source_id)
            except SourceRegistryError as exc:
                raise SeedManifestError(
                    f"source {source_id!r}: registry record malformed: {exc}"
                ) from exc
            if existing.source_class != source_class:
                raise SeedManifestError(
                    f"source {source_id!r}: class mismatch with existing "
                    f"registry record (recorded {existing.source_class!r}, "
                    f"manifest declares {source_class!r}); raw evidence "
                    "is immutable"
                )
            new_hash = SourceRegistry.hash_file(raw_path)
            if existing.content_hash != new_hash:
                raise SeedManifestError(
                    f"source {source_id!r}: raw evidence hash changed "
                    f"(recorded {existing.content_hash}, observed "
                    f"{new_hash}); raw evidence is immutable"
                )


# ---- Slice 076: report payload + workspace count snapshot -------------


def _snapshot_workspace_counts(workspace: Workspace) -> dict[str, int]:
    """Workspace-wide totals captured at the start / end of a real
    apply. The four buckets are intentionally minimal so the snapshot
    is cheap to compute and stable across slices:

    - ``sources`` — registered source records under
      ``state/source_registry/``.
    - ``entities`` — entity YAML files under ``claims/entities/``.
    - ``claims`` — total ``Assertion`` count across every entity
      container.
    - ``pages`` — Markdown files under ``pages/`` (recursive). The
      page tree is human-and-render-authored, so the count is the
      most operator-meaningful page total available without a full
      render walk.
    """
    registry = SourceRegistry(workspace)
    store = ClaimStore(workspace)
    sources = len(registry.list_ids())
    entity_ids = store.list_entity_ids()
    entities = len(entity_ids)
    claims = 0
    for eid in entity_ids:
        entity = store.load_entity(eid)
        claims += len(entity.assertions)
    pages = 0
    if workspace.pages.is_dir():
        pages = sum(1 for p in workspace.pages.rglob("*.md") if p.is_file())
    return {
        "sources": sources,
        "entities": entities,
        "claims": claims,
        "pages": pages,
    }


def _build_update_report(
    *,
    workspace: Workspace,
    manifest_relpath: str,
    op_ids: list[str],
    parsed_sources: list[dict[str, Any]],
    sources_planned: list[str],
    claims_planned: list[PlannedSeedClaim],
    claims_created: list[CreatedClaim],
    entities_touched: list[str],
    pages_rendered: list[str],
    no_render: bool,
    excerpt_checks: list[SeedExcerptCheck],
    counts_before: dict[str, int],
    counts_after: dict[str, int],
) -> dict[str, Any]:
    """Assemble the YAML report payload for a successful real apply.

    Manifest and source bytes are hashed via SHA-256 (matching the
    source registry's hash format). The report intentionally never
    embeds raw source bodies — excerpt previews flow through the
    bounded :class:`SeedExcerptCheck.excerpt_preview` field only.
    """
    manifest_abs = (workspace.root / manifest_relpath)
    if manifest_abs.is_file():
        manifest_sha = hash_file_sha256(manifest_abs)
    else:
        # Manifest path may be absolute / outside workspace; fall back
        # to the original absolute path the caller provided so the
        # report still records a hash. ``apply_seed_manifest``
        # preflights manifest existence, so this branch is reachable
        # only when the manifest was stored outside the workspace.
        manifest_sha = hash_file_sha256(Path(manifest_relpath))

    sources_payload: list[dict[str, Any]] = []
    for source in parsed_sources:
        source_id = source["source_id"]
        raw_path: Path = source["_resolved_path"]
        source_sha = hash_file_sha256(raw_path)
        planned_ids = [
            c.claim_id for c in claims_planned if c.source_id == source_id
        ]
        planned_id_set = set(planned_ids)
        created_ids_for_source = [
            c.claim_id for c in claims_created if c.claim_id in planned_id_set
        ]
        sources_payload.append(
            {
                "source_id": source_id,
                "source_class": source["source_class"],
                "path": source["path"],
                "sha256": source_sha,
                "planned_claim_ids": planned_ids,
                "created_claim_ids": created_ids_for_source,
            }
        )

    planned_payload = [
        {
            "source_id": c.source_id,
            "claim_id": c.claim_id,
            "entity_id": c.entity_id,
            "status": c.status,
        }
        for c in claims_planned
    ]
    # Include the effective excerpt_equality mode on each planned
    # entry so the report records what the manifest opted into per
    # claim, even when the mode was "none".
    for entry in planned_payload:
        for source in parsed_sources:
            if source["source_id"] != entry["source_id"]:
                continue
            for claim in source["claims"]:
                if claim["claim_id"] == entry["claim_id"]:
                    entry["excerpt_equality"] = claim.get(
                        _EXCERPT_EQUALITY_FIELD, "none"
                    )
                    break
            break

    created_payload = [
        {
            "claim_id": c.claim_id,
            "entity_id": c.entity_id,
            "status": c.status,
            "verification_status": c.verification_status,
        }
        for c in claims_created
    ]

    pages_payload = {
        "planned": sorted({p for p in pages_rendered}) if not no_render else [],
        "rendered": list(pages_rendered),
        "skipped": bool(no_render),
    }

    excerpt_checks_payload = [
        excerpt_check_to_mapping(c) for c in excerpt_checks
    ]

    return {
        "version": SEED_UPDATE_REPORT_VERSION,
        "op_id": op_ids[0],
        "op_ids": list(op_ids),
        "created_at": iso_now(),
        "manifest": {
            "path": manifest_relpath,
            "sha256": manifest_sha,
        },
        "provenance": {
            "generation_mode": "deterministic_seed",
            "model_provider": None,
            "provider_calls": 0,
            "api_cost_usd": 0,
        },
        "sources": sources_payload,
        "claims_planned": planned_payload,
        "claims_created": created_payload,
        "entities_touched": list(entities_touched),
        "pages": pages_payload,
        "counts": {
            "before": dict(counts_before),
            "after": dict(counts_after),
        },
        "excerpt_checks": excerpt_checks_payload,
        "refusal_reason": None,
        "errors": [],
    }
