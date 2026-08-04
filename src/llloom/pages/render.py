"""Claim-block rendering for variant-(B) pages.

Rendering is deterministic template expansion over claim records. The
LLM is not invoked at render time; this keeps render trivially
idempotent and testable.

Slice 071 moved the rendering surface from entity-centric (one render
call per ``(page_id, entity)``) to page/block-centric: every page is
rendered at most once per operation, and the rendered claim-block
region is the union of every entity's render-visible assertions
targeting that page's ``(page_id, block_id)``. Ordering is
deterministic by ``(entity_id, claim_id)`` so output is byte-stable
across re-runs and ingest orderings. Single-entity callers see the
exact same byte output as before; multi-entity pages get one
``## display_name`` section per contributing entity in
``entity_id`` order.

Contracts (from ``04_specification/component_contracts.md`` Â§Renderer):

- renderer writes only inside the claim-block region;
- commentary region survives byte-for-byte (renderer never reads
  commentary content);
- missing or malformed page markers are hard refusal conditions;
- non-idempotent render drift is a failure mode;
- render is deterministic template expansion: no ``LLMInvoke``, no
  model provider, no nondeterministic generator enters the render
  path.

Render-visible status filter (renderer-only — not the same as
graph/query active/inactive lifecycle filtering):

- excluded from rendered claim blocks:
  ``retracted``, ``retracted_by_source``, ``archived``;
- still rendered with the legacy marker shape:
  ``draft`` (``[DRAFT]``), ``reviewed`` (``[REVIEWED]``),
  ``validated`` (``[VALIDATED]``), ``stale`` (``[STALE]``),
  ``superseded`` (``[SUPERSEDED]``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from llloom.claims.models import Assertion, EntityContainer
from llloom.pages.regions import PageParseError, ParsedPage, parse_page, replace_claim_block
from llloom.workspace.layout import Workspace


class RenderError(Exception):
    """Raised when a page cannot be rendered."""


@dataclass
class RenderResult:
    page_id: str
    page_path: Path
    wrote: bool
    fingerprint: str
    rendered_claim_ids: list[str]


_RENDER_HIDDEN_STATUSES: frozenset[str] = frozenset(
    {"retracted", "retracted_by_source", "archived"}
)


@dataclass(frozen=True)
class _BlockContributor:
    """Internal: one entity's render-visible contribution to a block.

    Built by :func:`_select_contributors` from a list of canonical
    :class:`EntityContainer` instances and a target ``block_id``. The
    ``assertions`` tuple is filtered to render-visible statuses and
    sorted by ``claim_id`` for deterministic output.
    """

    entity_id: str
    display_name: str
    assertions: tuple[Assertion, ...]


def _select_contributors(
    entities: Iterable[EntityContainer],
    block_id: str,
) -> list[_BlockContributor]:
    """Render-visible filter + ``(entity_id, claim_id)`` sort.

    Entities with no render-visible assertion targeting ``block_id``
    are dropped. Entities that pass the filter contribute an
    ``_BlockContributor`` whose ``assertions`` are sorted by
    ``claim_id``. The returned list is then sorted by ``entity_id``.
    """
    out: list[_BlockContributor] = []
    for entity in entities:
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
            _BlockContributor(
                entity_id=entity.entity_id,
                display_name=entity.display_name,
                assertions=tuple(visible),
            )
        )
    out.sort(key=lambda c: c.entity_id)
    return out


def render_claim_block(
    entity: EntityContainer,
    block_id: str,
) -> str:
    """Single-entity claim-block renderer (legacy compatibility).

    Delegates to :func:`render_claim_block_from_contributors` with a
    one-element contributor list so output stays byte-identical to
    the pre-Slice-071 shape for single-entity pages with at least one
    render-visible assertion. The empty single-entity case (Slice
    071a follow-up) restores the pre-Slice-071 sentinel shape
    ``"## <display_name>\\n\\n_No rendered assertions._\\n"`` because
    the wrapper has unambiguous single-entity context — the union
    helper itself cannot pick a heading when zero contributors remain
    after filtering, so it returns the bare sentinel.
    """
    text = render_claim_block_from_contributors([entity], block_id)
    if text == "_No rendered assertions._\n":
        return f"## {entity.display_name}\n\n_No rendered assertions._\n"
    return text


def render_claim_block_from_contributors(
    contributors: Iterable[EntityContainer],
    block_id: str,
) -> str:
    """Render the union claim block from every contributor's
    render-visible assertions targeting ``block_id``.

    Output shape:

    - zero render-visible contributors: ``"_No rendered assertions._\\n"``;
    - one or more contributors: one ``## display_name`` section per
      entity in ``entity_id`` order, each containing the entity's
      sorted claim bullets.

    The single-contributor case is byte-identical to the legacy
    ``render_claim_block(entity, block_id)`` output: one heading,
    blank line, bullet list, trailing blank line.
    """
    selected = _select_contributors(contributors, block_id)
    if not selected:
        return "_No rendered assertions._\n"
    sections: list[str] = []
    for contributor in selected:
        lines: list[str] = [f"## {contributor.display_name}", ""]
        for a in contributor.assertions:
            marker = _status_marker(a)
            text = a.claim_text.strip().replace("\n", " ")
            lines.append(f"- {marker}{text} [claim:{a.claim_id}]")
        lines.append("")
        sections.append("\n".join(lines))
    return "\n".join(sections)


def compute_render_fingerprint(
    entity: EntityContainer,
    block_id: str,
) -> str:
    """Single-entity fingerprint (legacy compatibility).

    Delegates to :func:`compute_render_fingerprint_from_contributors`
    so callers that still hash one entity's contribution see the
    same byte format the multi-entity path uses. After Slice 071 the
    canonical workspace fingerprint stored under ``page_id`` is the
    union fingerprint, so single-entity pages now match
    ``compute_page_render_fingerprints(...)[page_id]`` exactly.
    """
    return compute_render_fingerprint_from_contributors([entity], block_id)


def compute_render_fingerprint_from_contributors(
    contributors: Iterable[EntityContainer],
    block_id: str,
) -> str:
    """Deterministic hash covering every render-visible assertion in
    ``contributors`` that targets ``block_id``.

    Inputs are sorted by ``(entity_id, claim_id)`` before hashing.
    Identical contributor sets — regardless of ingest order — produce
    identical hashes.
    """
    h = hashlib.sha256()
    h.update(f"block_id={block_id}\n".encode("utf-8"))
    selected = _select_contributors(contributors, block_id)
    for c in selected:
        h.update(b"---\n")
        h.update(f"entity_id={c.entity_id}\n".encode("utf-8"))
        h.update(f"display_name={c.display_name}\n".encode("utf-8"))
        for a in c.assertions:
            h.update(b"===\n")
            h.update(f"claim_id={a.claim_id}\n".encode("utf-8"))
            h.update(f"status={a.status}\n".encode("utf-8"))
            h.update(f"verification_status={a.verification_status}\n".encode("utf-8"))
            h.update(f"text={a.claim_text.strip()}\n".encode("utf-8"))
            for ev in a.evidence:
                h.update(f"ev={ev.source_id}:{ev.excerpt_hash}\n".encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


def render_page_file(
    workspace: Workspace,
    page_path: Path,
    entity: EntityContainer,
    dry_run: bool = False,
) -> RenderResult:
    """Render one on-disk page file against a single entity (legacy
    compatibility wrapper).

    Behaves like :func:`render_page_file_from_contributors` with a
    one-element contributor list, but routes the claim-block render
    through :func:`render_claim_block` so the empty single-entity
    page preserves the pre-Slice-071 sentinel shape
    ``"## <display_name>\\n\\n_No rendered assertions._\\n"`` (Slice
    071a follow-up). The fingerprint and ``rendered_claim_ids`` are
    computed via the union helpers so all six fingerprint-aware
    surfaces (`render`, `ingest`, `reconcile`, `retract`, `rebuild`,
    `lint`) still agree on the hash for a single-entity page.
    """
    if not page_path.is_file():
        raise RenderError(f"page file not found: {page_path}")
    original = page_path.read_text(encoding="utf-8")
    try:
        parsed: ParsedPage = parse_page(original)
    except PageParseError as exc:
        raise RenderError(f"{page_path}: {exc}") from exc

    new_inner = render_claim_block(entity, parsed.claim_block_id)
    new_text = replace_claim_block(parsed, new_inner)
    selected = _select_contributors([entity], parsed.claim_block_id)
    rendered_ids: list[str] = [
        a.claim_id for c in selected for a in c.assertions
    ]
    fingerprint = compute_render_fingerprint_from_contributors(
        [entity], parsed.claim_block_id
    )
    wrote = new_text != original
    if wrote and not dry_run:
        _atomic_write_text(page_path, new_text)
    page_id = parsed.frontmatter.get("page_id", page_path.stem)
    return RenderResult(
        page_id=str(page_id),
        page_path=page_path,
        wrote=wrote,
        fingerprint=fingerprint,
        rendered_claim_ids=rendered_ids,
    )


def render_page_file_from_contributors(
    workspace: Workspace,
    page_path: Path,
    contributors: Iterable[EntityContainer],
    dry_run: bool = False,
) -> RenderResult:
    """Render one on-disk page file against the union of all
    contributors' render-visible assertions.

    Reads the page bytes, parses markers, renders the claim block
    from every contributor whose assertions target the parsed
    ``block_id``, and writes the result atomically. Preserves the
    commentary region byte-for-byte via :func:`replace_claim_block`.
    Variant-(B) is preserved: exactly one claim-block region and one
    commentary region survive the rewrite.
    """
    if not page_path.is_file():
        raise RenderError(f"page file not found: {page_path}")
    original = page_path.read_text(encoding="utf-8")
    try:
        parsed: ParsedPage = parse_page(original)
    except PageParseError as exc:
        raise RenderError(f"{page_path}: {exc}") from exc

    contributors_list = list(contributors)
    new_inner = render_claim_block_from_contributors(
        contributors_list, parsed.claim_block_id
    )
    new_text = replace_claim_block(parsed, new_inner)
    selected = _select_contributors(contributors_list, parsed.claim_block_id)
    rendered_ids: list[str] = [
        a.claim_id for c in selected for a in c.assertions
    ]
    fingerprint = compute_render_fingerprint_from_contributors(
        contributors_list, parsed.claim_block_id
    )
    wrote = new_text != original
    if wrote and not dry_run:
        _atomic_write_text(page_path, new_text)
    page_id = parsed.frontmatter.get("page_id", page_path.stem)
    return RenderResult(
        page_id=str(page_id),
        page_path=page_path,
        wrote=wrote,
        fingerprint=fingerprint,
        rendered_claim_ids=rendered_ids,
    )


def compute_page_render_fingerprints(
    entities: Iterable[EntityContainer],
) -> dict[str, str]:
    """Compute the union render fingerprint for every page touched by
    any assertion's render_targets across ``entities``.

    Returns a ``dict[page_id, fingerprint]``. Each fingerprint is the
    union over all entities whose render_targets reach that page. Used
    by ``lint`` (stale-fingerprint detection), ``rebuild
    render_fingerprints``, the health-report drift detection, and
    ``reconcile`` so every fingerprint-aware surface sees the same
    page/block-centric value the renderer writes.
    """
    entities_list = list(entities)
    page_to_block_id: dict[str, str] = {}
    page_to_contributors: dict[str, list[EntityContainer]] = {}
    for entity in entities_list:
        seen_pages_for_entity: set[str] = set()
        for assertion in entity.assertions:
            for target in assertion.render_targets:
                # Variant-(B) holds at most one claim-block per page;
                # the first encountered block_id for a page wins as
                # canonical. Cross-block consistency would be a spec
                # violation upstream, not a renderer concern.
                page_to_block_id.setdefault(target.page_id, target.block_id)
                if target.page_id in seen_pages_for_entity:
                    continue
                seen_pages_for_entity.add(target.page_id)
                page_to_contributors.setdefault(target.page_id, []).append(entity)
    out: dict[str, str] = {}
    for page_id, contributors in page_to_contributors.items():
        contributors.sort(key=lambda e: e.entity_id)
        block_id = page_to_block_id[page_id]
        out[page_id] = compute_render_fingerprint_from_contributors(
            contributors, block_id
        )
    return out


def _status_marker(assertion: Assertion) -> str:
    if assertion.status == "stale":
        return "[STALE] "
    if assertion.status == "draft":
        return "[DRAFT] "
    if assertion.status == "validated":
        return "[VALIDATED] "
    if assertion.status == "reviewed":
        return "[REVIEWED] "
    if assertion.status == "superseded":
        return "[SUPERSEDED] "
    return ""


def resolve_page_path(workspace: Workspace, page_id: str) -> Path:
    """Map a page_id to its on-disk path.

    Tries ``pages/<page_id>.md`` first (so a page_id like
    ``concept/complementarity`` resolves to
    ``pages/concept/complementarity.md``). If that file does not exist,
    falls back to any of the class directories
    (``entities|concepts|syntheses|navigation``) using the trailing
    segment of the page_id. Returns the first existing path; otherwise
    returns the primary candidate (which may not exist).
    """
    primary = workspace.root / "pages" / f"{page_id}.md"
    if primary.is_file():
        return primary
    tail = page_id.split("/")[-1]
    for class_dir in ("entities", "concepts", "syntheses", "navigation"):
        alt = workspace.pages / class_dir / f"{tail}.md"
        if alt.is_file():
            return alt
    return primary


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
