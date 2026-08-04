"""Read-only inspection operations (Slice 077).

This module hosts the listing / card surface that makes authoritative
state visible during daily use:

- :func:`list_claims` — filtered, deterministic claim summaries;
- :func:`claim_card` — full inspection of one claim
  (``claim:<entity>:<claim>`` or a bare unique ``claim_id``);
- :func:`list_sources` — registry metadata only (no raw source text);
- :func:`list_pages` — page-frontmatter mirror;
- :func:`list_render_targets` — reuses the Slice 073 read-only render
  plan without acquiring any lock, opening any journal entry, or
  writing any page / fingerprint / transaction directory.

Every operation here is **read-only**: no lock, no journal entry, no
transaction directory, no page / fingerprint / claim / registry /
report / sidecar write, no model / provider invocation. The operations
walk canonical YAML and read-only registry / render APIs only.
"""

from __future__ import annotations

import re
from typing import Iterable

import yaml

from llloom.claims.models import (
    CLAIM_STATUSES,
    Assertion,
    EntityContainer,
    VERIFICATION_STATUSES,
)
from llloom.claims.store import ClaimStore
from llloom.ops.results import (
    ClaimCard,
    ClaimSummary,
    EvidenceSummary,
    PageSummary,
    RenderTargetListEntry,
    RenderTargetSummary,
    SourceSummary,
)
from llloom.pages.regions import PageParseError, parse_page
from llloom.pages.render import resolve_page_path
from llloom.sources.registry import SourceRegistry
from llloom.workspace.layout import Workspace


# Bounded preview for claim-text in list summaries and evidence-excerpt
# previews on the card. Matches the verifier / seed-report bound so
# inspection surfaces never carry raw source bodies.
_PREVIEW_MAX_CHARS = 240


class ClaimCardError(Exception):
    """Raised by :func:`claim_card` when the target is missing or
    ambiguous. The CLI catches this and prints a concise stderr
    diagnostic — ambiguity messages include the candidate qualified
    ids so the caller can disambiguate.
    """


class InspectFilterError(ValueError):
    """Raised by list operations when a filter argument is malformed
    (unknown lifecycle status, unknown verification status, empty
    value). The CLI catches this and prints a concise diagnostic.
    """


# ---- shared helpers ---------------------------------------------------


def _bounded_preview(text: str, *, max_chars: int = _PREVIEW_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _qualified_target(entity_id: str, claim_id: str) -> str:
    return f"claim:{entity_id}:{claim_id}"


def _resolve_status_set(
    value: str | Iterable[str] | None, *, label: str = "status"
) -> frozenset[str] | None:
    """Validate a status filter against ``CLAIM_STATUSES``. Returns
    ``None`` (no filter) for ``None``; otherwise a frozenset.
    ``"all"`` resolves to every state.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value == "all":
            return frozenset(CLAIM_STATUSES)
        if value not in CLAIM_STATUSES:
            raise InspectFilterError(
                f"unknown {label} {value!r}; allowed: "
                f"{sorted(CLAIM_STATUSES)} or 'all'"
            )
        return frozenset({value})
    seen: set[str] = set()
    for item in value:
        if item == "all":
            return frozenset(CLAIM_STATUSES)
        if item not in CLAIM_STATUSES:
            raise InspectFilterError(
                f"unknown {label} {item!r}; allowed: "
                f"{sorted(CLAIM_STATUSES)} or 'all'"
            )
        seen.add(item)
    if not seen:
        raise InspectFilterError(f"{label} filter must name at least one state")
    return frozenset(seen)


def _resolve_verification_set(
    value: str | Iterable[str] | None,
) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in VERIFICATION_STATUSES:
            raise InspectFilterError(
                f"unknown verification_status {value!r}; allowed: "
                f"{sorted(VERIFICATION_STATUSES)}"
            )
        return frozenset({value})
    seen: set[str] = set()
    for item in value:
        if item not in VERIFICATION_STATUSES:
            raise InspectFilterError(
                f"unknown verification_status {item!r}; allowed: "
                f"{sorted(VERIFICATION_STATUSES)}"
            )
        seen.add(item)
    if not seen:
        raise InspectFilterError(
            "verification_status filter must name at least one state"
        )
    return frozenset(seen)


def _resolve_string_set(
    value: str | Iterable[str] | None, *, label: str
) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            raise InspectFilterError(f"{label} filter must not be empty")
        return frozenset({value})
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item:
            raise InspectFilterError(
                f"{label} filter entries must be non-empty strings"
            )
        seen.add(item)
    if not seen:
        raise InspectFilterError(f"{label} filter must name at least one value")
    return frozenset(seen)


def _summarize_evidence(assertion: Assertion) -> list[EvidenceSummary]:
    out: list[EvidenceSummary] = []
    for ev in assertion.evidence:
        preview = None
        if ev.excerpt is not None:
            preview = _bounded_preview(ev.excerpt)
        out.append(
            EvidenceSummary(
                source_id=ev.source_id,
                locator=ev.locator.to_mapping(),
                excerpt_hash=ev.excerpt_hash,
                excerpt_preview=preview,
            )
        )
    return out


def _summarize_render_targets(assertion: Assertion) -> list[RenderTargetSummary]:
    return [
        RenderTargetSummary(page_id=rt.page_id, block_id=rt.block_id)
        for rt in assertion.render_targets
    ]


def _claim_summary(
    entity: EntityContainer, assertion: Assertion
) -> ClaimSummary:
    return ClaimSummary(
        qualified_target=_qualified_target(entity.entity_id, assertion.claim_id),
        claim_id=assertion.claim_id,
        entity_id=entity.entity_id,
        entity_display_name=entity.display_name,
        entity_type=entity.entity_type,
        claim_kind=assertion.claim_kind,
        status=assertion.status,
        verification_status=assertion.verification_status,
        supersedes=assertion.supersedes,
        source_ids=[ev.source_id for ev in assertion.evidence],
        render_targets=_summarize_render_targets(assertion),
        claim_text_preview=_bounded_preview(assertion.claim_text),
    )


def _claim_card(
    entity: EntityContainer, assertion: Assertion
) -> ClaimCard:
    return ClaimCard(
        qualified_target=_qualified_target(entity.entity_id, assertion.claim_id),
        claim_id=assertion.claim_id,
        entity_id=entity.entity_id,
        entity_display_name=entity.display_name,
        entity_type=entity.entity_type,
        claim_kind=assertion.claim_kind,
        claim_text=assertion.claim_text,
        status=assertion.status,
        verification_status=assertion.verification_status,
        supersedes=assertion.supersedes,
        evidence=_summarize_evidence(assertion),
        render_targets=_summarize_render_targets(assertion),
        created_at=assertion.created_at,
        updated_at=assertion.updated_at,
    )


# ---- list_claims ------------------------------------------------------


def list_claims(
    workspace: Workspace,
    *,
    status: str | Iterable[str] | None = None,
    verification_status: str | Iterable[str] | None = None,
    entity_id: str | Iterable[str] | None = None,
    role: str | Iterable[str] | None = None,
) -> list[ClaimSummary]:
    """Return every claim that matches the filters.

    ``status=None`` means "all states" for the listing surface
    (matches operator intent — listing should default to showing
    everything available). For query-time discoverability the
    default-broad rule lives in :func:`llloom.ops.query.query`.
    Pass ``status="all"`` explicitly to be unambiguous; the values
    behave identically.

    Ordering: stable on ``(entity_id, claim_id)`` so consecutive
    calls return byte-identical results on an unchanged workspace.

    Read-only: walks canonical entity YAML only. No lock, no
    journal, no model call.
    """
    status_filter = _resolve_status_set(status)
    verification_filter = _resolve_verification_set(verification_status)
    entity_filter = _resolve_string_set(entity_id, label="entity_id")
    role_filter = _resolve_string_set(role, label="role")

    store = ClaimStore(workspace)
    out: list[ClaimSummary] = []
    for entity in store.iter_entities():
        if entity_filter is not None and entity.entity_id not in entity_filter:
            continue
        for assertion in entity.assertions:
            if (
                status_filter is not None
                and assertion.status not in status_filter
            ):
                continue
            if (
                verification_filter is not None
                and assertion.verification_status not in verification_filter
            ):
                continue
            if (
                role_filter is not None
                and assertion.claim_kind not in role_filter
            ):
                continue
            out.append(_claim_summary(entity, assertion))
    out.sort(key=lambda s: (s.entity_id, s.claim_id))
    return out


# ---- claim_card -------------------------------------------------------


_QUALIFIED_TARGET_RE = re.compile(r"^claim:(?P<entity>[^:]+):(?P<claim>.+)$")


def claim_card(workspace: Workspace, target: str) -> ClaimCard:
    """Inspect one claim.

    ``target`` is either:

    - the qualified form ``claim:<entity_id>:<claim_id>``, or
    - a bare ``<claim_id>`` (only when exactly one claim across
      the whole workspace carries that id).

    Raises :class:`ClaimCardError` for: malformed targets, missing
    qualified targets, missing bare ids, and ambiguous bare ids
    (the ambiguity message lists every candidate qualified
    target).
    """
    if not isinstance(target, str) or not target:
        raise ClaimCardError("claim card target must be a non-empty string")

    store = ClaimStore(workspace)

    match = _QUALIFIED_TARGET_RE.match(target)
    if match is not None:
        entity_id = match.group("entity")
        claim_id = match.group("claim")
        if not store.exists(entity_id):
            raise ClaimCardError(f"entity not found: {entity_id!r}")
        entity = store.load_entity(entity_id)
        assertion = entity.find_assertion(claim_id)
        if assertion is None:
            raise ClaimCardError(
                f"claim {claim_id!r} not found on entity {entity_id!r}"
            )
        return _claim_card(entity, assertion)

    # Bare claim id — search every entity. Refuse on missing or
    # ambiguous ids with a structured diagnostic.
    candidates: list[tuple[EntityContainer, Assertion]] = []
    for entity in store.iter_entities():
        assertion = entity.find_assertion(target)
        if assertion is not None:
            candidates.append((entity, assertion))
    if not candidates:
        raise ClaimCardError(
            f"no claim with claim_id {target!r} found in the workspace; "
            f"use the qualified form claim:<entity>:<claim> to disambiguate"
        )
    if len(candidates) > 1:
        qualified_ids = [
            _qualified_target(e.entity_id, a.claim_id) for e, a in candidates
        ]
        qualified_ids.sort()
        raise ClaimCardError(
            f"bare claim_id {target!r} is ambiguous; matches "
            f"{len(candidates)} claims: {qualified_ids}. Use the "
            "qualified form claim:<entity>:<claim> to pick one."
        )
    entity, assertion = candidates[0]
    return _claim_card(entity, assertion)


# ---- list_sources -----------------------------------------------------


def list_sources(
    workspace: Workspace,
    *,
    status: str | Iterable[str] | None = None,
    source_class: str | Iterable[str] | None = None,
) -> list[SourceSummary]:
    """Read-only source-registry listing.

    Reports registry metadata only — never raw source text. The
    ``status`` filter accepts the registry's
    ``registered`` / ``retracted`` strings. ``source_class``
    filters by class string. Sorted by ``source_id``.
    """
    status_filter = _resolve_string_set(status, label="source status") if status is not None else None
    class_filter = _resolve_string_set(source_class, label="source_class")

    registry = SourceRegistry(workspace)
    out: list[SourceSummary] = []
    for record in registry.iter_records():
        if status_filter is not None and record.status not in status_filter:
            continue
        if class_filter is not None and record.source_class not in class_filter:
            continue
        out.append(
            SourceSummary(
                source_id=record.source_id,
                source_class=record.source_class,
                raw_path=record.raw_path,
                content_hash=record.content_hash,
                byte_size=record.byte_size,
                status=record.status,
                registered_at=record.registered_at,
                last_seen_at=record.last_seen_at,
                retracted_at=record.retracted_at,
                retraction_reason=record.retraction_reason,
            )
        )
    out.sort(key=lambda s: s.source_id)
    return out


# ---- list_pages -------------------------------------------------------


def list_pages(workspace: Workspace) -> list[PageSummary]:
    """Read-only page tree listing.

    Walks ``pages/`` recursively for ``*.md`` files. For each page
    the frontmatter is parsed leniently — if the frontmatter parser
    fails, the page is still listed (the ``page_id`` falls back to
    the file stem and the frontmatter fields stay empty) so a
    broken page never silently drops out of the listing.

    Returns entries sorted by workspace-relative page path so
    consecutive calls are byte-stable.
    """
    out: list[PageSummary] = []
    pages_root = workspace.pages
    if not pages_root.is_dir():
        return out
    for path in sorted(pages_root.rglob("*.md")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        page_id = path.stem
        page_class = ""
        status = ""
        write_policy = ""
        try:
            parsed = parse_page(text)
            frontmatter = parsed.frontmatter or {}
        except PageParseError:
            frontmatter = _try_parse_frontmatter(text)
        if isinstance(frontmatter, dict):
            page_id = str(frontmatter.get("page_id") or page_id)
            page_class = str(frontmatter.get("page_class") or "")
            status = str(frontmatter.get("status") or "")
            write_policy = str(frontmatter.get("write_policy") or "")
        rel = path.resolve().relative_to(workspace.root.resolve()).as_posix()
        out.append(
            PageSummary(
                page_id=page_id,
                page_path=rel,
                page_class=page_class,
                status=status,
                write_policy=write_policy,
            )
        )
    out.sort(key=lambda p: p.page_path)
    return out


def _try_parse_frontmatter(text: str) -> dict | None:
    """Best-effort YAML frontmatter parser for pages that fail the
    strict claim-block / commentary marker contract. Used only by
    :func:`list_pages` so a broken page still appears in the
    listing.
    """
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    block = text[3:end].lstrip("\n")
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError:
        return None
    if isinstance(data, dict):
        return data
    return None


# ---- list_render_targets ----------------------------------------------


def list_render_targets(
    workspace: Workspace,
    *,
    page: str | None = None,
) -> list[RenderTargetListEntry]:
    """Read-only render-target enumeration.

    Reuses the Slice 073 ``render(..., list_targets=True)`` plan
    without acquiring the workspace lock, opening a render
    journal entry, or writing any page / fingerprint /
    transaction directory. ``page`` (optional) limits the listing
    to one ``page_id``. Sorted by ``page_id``.
    """
    # Import lazily to avoid a circular dependency between
    # ``ops.inspect`` and ``ops.render`` at module load time.
    from llloom.ops.render import render

    target = f"page:{page}" if page is not None else None
    plan_result = render(workspace, target=target, list_targets=True)
    out: list[RenderTargetListEntry] = []
    for entry in plan_result.plan:
        # Materialize the workspace-relative page path even when
        # the plan entry stores it on the rendered output side.
        if entry.page_path:
            page_path = entry.page_path
        else:
            page_path = resolve_page_path(workspace, entry.page_id).name
        out.append(
            RenderTargetListEntry(
                page_id=entry.page_id,
                block_id=entry.block_id,
                page_path=page_path,
                marker_health=entry.marker_health,
                marker_message=entry.marker_message,
                contributing_claim_ids=list(entry.contributing_claim_ids),
            )
        )
    out.sort(key=lambda e: e.page_id)
    return out
