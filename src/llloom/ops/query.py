"""`query` operation (read-only).

Two retrieval paths, both deterministic and citation-first:

1. **Authoritative claim retrieval.** Substring/token match against
   claim text on every non-retracted assertion in the workspace.
2. **Strict-`index_only` verbatim retrieval.** For sources whose
   resolved ingest policy is ``index_only``, walk the raw source text
   and return bounded verbatim spans whose surrounding window
   contains a question token. The raw source body is never passed to
   ``LLMInvoke``: ``query`` does not invoke the harness in the first
   slice, and any future invocation would receive only the bounded
   ``SourceSpan`` typed input, never a ``SourceDocument``.

If the optional search sidecar (``state/search/search.sqlite``) is
present, it may select candidate ids for either path, but every
emitted citation or verbatim span is still rehydrated from canonical
claim YAML or the raw registered source file. The sidecar is never
canonical and may be deleted without data loss.

No free prose synthesis. No answer filing. No claim creation. The
answer string is a deterministic textual rendering of the claim and
span citations.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import yaml

from llloom.claims.models import CLAIM_STATUSES, VERIFICATION_STATUSES
from llloom.claims.store import ClaimStore, ClaimStoreError
from llloom.ops.results import QueryResult, StructureItemHit, VerbatimSpan
from llloom.schema.policy import load_schema
from llloom.sources.registry import SourceRegistry, SourceRegistryError
from llloom.state.search import SearchHit, search_candidates, sidecar_exists
from llloom.workspace.layout import Workspace


# Slice 077: the legacy default-query status filter — the set of
# lifecycle states that the broad ``status=None`` mode silently skips.
# Kept identical to the pre-Slice-077 behavior so the default API
# is byte-compatible. ``status="all"`` opts out of this filter so
# the broad inspection mode can surface every state.
_DEFAULT_HIDDEN_STATUSES: frozenset[str] = frozenset(
    {"retracted", "retracted_by_source", "archived", "stale", "superseded"}
)

# Bounded preview length for any citation's claim text. Keeping
# citations short matches the existing ``_SNIPPET_RADIUS`` story and
# avoids accidentally surfacing large source-grounded text into a
# JSON result that may flow back to model contexts.
_CITATION_TEXT_MAX_CHARS = 240


class QueryFilterError(ValueError):
    """Raised by ``query`` (and the CLI dispatcher) when a filter
    argument is malformed — unknown lifecycle status, unknown
    verification status, or an empty value. The CLI catches this
    and prints a concise stderr diagnostic instead of a traceback.
    """


# Bounded snippet around a token match. Window is in characters and is
# snapped to whitespace boundaries so we never split a word in the
# middle. Keeping this small and deterministic avoids leaking
# unintended context (canary tokens, neighbouring paragraphs).
_SNIPPET_RADIUS = 120
_MAX_SPANS_TOTAL = 5
_MAX_SPANS_PER_SOURCE = 3
_MAX_STRUCTURE_ITEMS_TOTAL = 5


def query(
    workspace: Workspace,
    *,
    question: str,
    max_citations: int = 5,
    status: str | Iterable[str] | None = None,
    verification_status: str | Iterable[str] | None = None,
    entity_id: str | Iterable[str] | None = None,
    role: str | Iterable[str] | None = None,
    ids_only: bool = False,
) -> QueryResult:
    """Read-only authoritative query with optional filters.

    Slice 077 added the four filter knobs (``status``,
    ``verification_status``, ``entity_id``, ``role``) and the
    ``ids_only`` flag. None of them widen the visible state — every
    filter is an AND constraint on top of the token-match retrieval
    and the canonical status-skip rule from the original V1 query.

    ``status``:

    - ``None`` (default) — preserve the pre-Slice-077 broad behavior:
      skip ``retracted``, ``retracted_by_source``, ``archived``,
      ``stale``, ``superseded``;
    - ``"all"`` — opt explicitly into showing every lifecycle state
      (the "broad inspection" mode);
    - one status string, or an iterable of status strings — limit to
      those states. Each value is validated against ``CLAIM_STATUSES``;
      unknown values raise :class:`QueryFilterError`.

    ``verification_status``: ``None`` (no filter) or one or more
    values from ``VERIFICATION_STATUSES``.

    ``entity_id``: ``None`` (no filter) or one or more canonical
    entity ids; only claims on those entities are eligible.

    ``role``: ``None`` (no filter) or one or more values matching
    :attr:`Assertion.claim_kind`. ``claim_kind`` is the V1 role/kind
    field — the codebase has no separate ``role`` attribute on
    claims, so the filter delegates to ``claim_kind`` directly. The
    decision is documented in the slice spec.

    ``ids_only``: when ``True`` the answer string is empty, no
    verbatim spans are computed, and no structure-item rehydration
    runs. ``used_claim_ids`` carries the filtered + ranked claim
    ids. The shape is identical to the broad path apart from these
    fields, so CLI / library callers can switch back and forth
    without a separate code path. The CLI ``--ids-only`` flag turns
    this on and emits one id per line.
    """
    status_filter = _resolve_status_filter(status)
    verification_filter = _resolve_verification_filter(verification_status)
    entity_filter = _resolve_id_set(entity_id, label="entity_id")
    role_filter = _resolve_id_set(role, label="role")

    store = ClaimStore(workspace)
    tokens = _tokenize(question)

    # Candidate hints from the FTS5 sidecar, or None if absent. The
    # sidecar may narrow the set we scan but never expands what can
    # appear in citations or spans: rehydration below still walks the
    # canonical records and re-applies status filters.
    hints = _sidecar_hints(workspace, question) if sidecar_exists(workspace) else None

    citations, used_claim_ids = _retrieve_claims(
        store=store,
        tokens=tokens,
        max_citations=max_citations,
        claim_hints=None if hints is None else hints["claim_ids"],
        status_filter=status_filter,
        verification_filter=verification_filter,
        entity_filter=entity_filter,
        role_filter=role_filter,
        ids_only=ids_only,
    )

    if ids_only:
        return QueryResult(
            question=question,
            answer="",
            citations=citations,
            used_claim_ids=used_claim_ids,
            used_verbatim_spans=[],
            used_structure_items=[],
            ids_only=True,
        )

    verbatim_spans = _retrieve_index_only_spans(
        workspace=workspace,
        tokens=tokens,
        source_hints=None if hints is None else hints["source_ids"],
    )

    structure_items = _retrieve_structure_items(
        workspace=workspace,
        structure_hints=None if hints is None else hints["structure_hits"],
    )

    answer = _format_answer(citations, verbatim_spans, structure_items)
    return QueryResult(
        question=question,
        answer=answer,
        citations=citations,
        used_claim_ids=used_claim_ids,
        used_verbatim_spans=verbatim_spans,
        used_structure_items=structure_items,
        ids_only=False,
    )


def _resolve_status_filter(
    value: str | Iterable[str] | None,
) -> frozenset[str] | None:
    """Translate the public ``status`` argument into either:

    - ``None`` — apply the default-broad rule (skip the
      ``_DEFAULT_HIDDEN_STATUSES`` set);
    - a (possibly empty) ``frozenset[str]`` of allowed states —
      ``"all"`` resolves to the full ``CLAIM_STATUSES`` set; a
      single string or any iterable of strings is validated against
      ``CLAIM_STATUSES`` and returned as a frozenset.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value == "all":
            return frozenset(CLAIM_STATUSES)
        if value not in CLAIM_STATUSES:
            raise QueryFilterError(
                f"unknown lifecycle status {value!r}; allowed: "
                f"{sorted(CLAIM_STATUSES)} or 'all'"
            )
        return frozenset({value})
    seen: set[str] = set()
    for item in value:
        if item == "all":
            return frozenset(CLAIM_STATUSES)
        if item not in CLAIM_STATUSES:
            raise QueryFilterError(
                f"unknown lifecycle status {item!r}; allowed: "
                f"{sorted(CLAIM_STATUSES)} or 'all'"
            )
        seen.add(item)
    if not seen:
        raise QueryFilterError("status filter must name at least one state")
    return frozenset(seen)


def _resolve_verification_filter(
    value: str | Iterable[str] | None,
) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in VERIFICATION_STATUSES:
            raise QueryFilterError(
                f"unknown verification_status {value!r}; allowed: "
                f"{sorted(VERIFICATION_STATUSES)}"
            )
        return frozenset({value})
    seen: set[str] = set()
    for item in value:
        if item not in VERIFICATION_STATUSES:
            raise QueryFilterError(
                f"unknown verification_status {item!r}; allowed: "
                f"{sorted(VERIFICATION_STATUSES)}"
            )
        seen.add(item)
    if not seen:
        raise QueryFilterError(
            "verification_status filter must name at least one state"
        )
    return frozenset(seen)


def _resolve_id_set(
    value: str | Iterable[str] | None, *, label: str
) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            raise QueryFilterError(f"{label} filter must not be empty")
        return frozenset({value})
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item:
            raise QueryFilterError(
                f"{label} filter entries must be non-empty strings"
            )
        seen.add(item)
    if not seen:
        raise QueryFilterError(f"{label} filter must name at least one value")
    return frozenset(seen)


def _sidecar_hints(workspace: Workspace, question: str) -> dict | None:
    """Return ``{"claim_ids": set, "source_ids": set,
    "structure_hits": list[SearchHit]}`` from the sidecar, or
    ``None`` if the sidecar returned nothing useful. The sidecar is
    advisory; if it is empty or fails to open we fall back to the full
    canonical walk rather than returning zero results.
    """
    try:
        hits = search_candidates(workspace, question)
    except Exception:
        return None
    if not hits:
        return None
    claim_ids = {h.claim_id for h in hits if h.doc_type == "claim" and h.claim_id}
    source_ids = {
        h.source_id for h in hits if h.doc_type == "index_only_source" and h.source_id
    }
    structure_hits = [h for h in hits if h.doc_type == "structure_item"]
    if not claim_ids and not source_ids and not structure_hits:
        return None
    return {
        "claim_ids": claim_ids,
        "source_ids": source_ids,
        "structure_hits": structure_hits,
    }


# ---- claim retrieval ---------------------------------------------------


def _retrieve_claims(
    *,
    store: ClaimStore,
    tokens: list[str],
    max_citations: int,
    claim_hints: set[str] | None = None,
    status_filter: frozenset[str] | None = None,
    verification_filter: frozenset[str] | None = None,
    entity_filter: frozenset[str] | None = None,
    role_filter: frozenset[str] | None = None,
    ids_only: bool = False,
) -> tuple[list[dict], list[str]]:
    """Walk claim YAMLs and score assertions. When ``claim_hints`` is
    provided (from the sidecar), claim ids it does not name receive a
    weaker score — the canonical text still controls whether a hit is
    emitted, but sidecar-preferred ids rank higher for tie-breaking.

    Slice 077 filter semantics:

    - ``status_filter is None`` — apply the pre-Slice-077
      default-skip set (``_DEFAULT_HIDDEN_STATUSES``);
    - ``status_filter`` is a frozenset of allowed lifecycle
      states — only assertions whose ``status`` is in the set
      are eligible;
    - ``verification_filter`` / ``entity_filter`` / ``role_filter``
      are AND constraints with the text query and with each
      other. The sidecar is advisory and cannot bypass these
      filters: a sidecar-hinted ``claim_id`` whose canonical
      record fails any filter is dropped here.

    Slice 077a empty-token compatibility (the load-bearing
    cleanup): an empty token list (``tokens == []``) is only
    treated as "list every claim that passes the filters" when
    the caller **opted into inspection** by passing a filter
    knob or ``ids_only=True``. The plain default
    ``query(workspace, question="")`` keeps the pre-Slice-077
    behavior — no citations, no used ids — because the token
    scorer returns ``0`` and the canonical ``score > 0``
    requirement holds. ``ids_only=True`` is treated as
    explicit inspection because it is a deliberate, non-default
    kwarg whose only useful meaning on an empty question is
    "give me the ids" — requiring a separate filter would force
    every shell user to add ``--status all`` to get the obvious
    behavior.
    """
    explicit_inspection = (
        status_filter is not None
        or verification_filter is not None
        or entity_filter is not None
        or role_filter is not None
        or ids_only
    )
    hits: list[tuple[int, str, str, str, "object"]] = []
    for entity in store.iter_entities():
        if entity_filter is not None and entity.entity_id not in entity_filter:
            continue
        for assertion in entity.assertions:
            if status_filter is None:
                if assertion.status in _DEFAULT_HIDDEN_STATUSES:
                    continue
            else:
                if assertion.status not in status_filter:
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

            score = _score(tokens, assertion.claim_text)
            hinted = claim_hints is not None and assertion.claim_id in claim_hints
            if hinted:
                # Sidecar rescue: FTS5 tokenization (unicode-fold, no
                # punctuation literals) can surface claims the naive
                # substring scorer misses. The rehydrated canonical
                # claim_text + filter checks above still control
                # acceptance; an FTS hit on a filtered-out claim was
                # already dropped before this point.
                score = max(score, 1)
            # Slice 077a: empty-token admission requires explicit
            # inspection (any Slice 077 filter knob or ids_only).
            # Otherwise the plain default empty-query path keeps the
            # pre-Slice-077 behavior of returning no citations.
            if score > 0 or (not tokens and explicit_inspection):
                hits.append(
                    (score, entity.entity_id, assertion.claim_id, assertion.claim_text, assertion)
                )
    hits.sort(key=lambda h: h[0], reverse=True)
    top = hits[:max_citations]

    citations: list[dict] = []
    used: list[str] = []
    for _, entity_id, claim_id, text, assertion in top:
        citations.append(_citation_dict(entity_id, assertion, text))
        used.append(claim_id)
    return citations, used


def _citation_dict(entity_id: str, assertion: Any, text: str) -> dict[str, Any]:
    """Slice 077: foreground authority metadata on each citation.

    Each citation now includes ``entity_id``, ``claim_id``,
    ``text`` / ``claim_text`` (preserved for back-compat),
    ``status``, ``verification_status``, ``claim_kind``,
    ``supersedes`` (when present), ``source_ids`` (deduplicated
    from the assertion's evidence list), and ``render_targets``
    (list of ``{page_id, block_id}`` mappings). The ``text``
    field stays under :data:`_CITATION_TEXT_MAX_CHARS` so a
    long claim does not balloon the citation payload.
    """
    source_ids: list[str] = []
    for ev in assertion.evidence:
        if ev.source_id not in source_ids:
            source_ids.append(ev.source_id)
    render_targets: list[dict[str, str]] = [
        {"page_id": rt.page_id, "block_id": rt.block_id}
        for rt in assertion.render_targets
    ]
    preview = text if len(text) <= _CITATION_TEXT_MAX_CHARS else text[: _CITATION_TEXT_MAX_CHARS - 3] + "..."
    out: dict[str, Any] = {
        "entity_id": entity_id,
        "claim_id": assertion.claim_id,
        "text": preview,
        "claim_text": preview,
        "claim_kind": assertion.claim_kind,
        "status": assertion.status,
        "verification_status": assertion.verification_status,
        "source_ids": source_ids,
        "render_targets": render_targets,
    }
    if assertion.supersedes is not None:
        out["supersedes"] = assertion.supersedes
    return out


# ---- index_only verbatim retrieval -------------------------------------


def _retrieve_index_only_spans(
    *,
    workspace: Workspace,
    tokens: list[str],
    source_hints: set[str] | None = None,
) -> list[VerbatimSpan]:
    """Walk registered ``index_only`` sources and return bounded spans.

    When ``source_hints`` is provided (from the sidecar), we prefer
    those sources first; others are still walked so a stale or empty
    sidecar cannot shrink the answer set below the canonical truth.
    Retracted sources and sources missing on disk are skipped; a
    sidecar row pointing at either never reaches ``QueryResult``.
    """
    if not tokens:
        return []
    schema = load_schema(workspace)
    registry = SourceRegistry(workspace)

    records = []
    for record in registry.iter_records():
        if record.status == "retracted":
            continue
        try:
            policy = schema.resolve_ingest_policy(record.source_class)
        except Exception:
            continue
        if policy != "index_only":
            continue
        records.append(record)

    if source_hints is not None:
        records.sort(key=lambda r: 0 if r.source_id in source_hints else 1)

    out: list[VerbatimSpan] = []
    for record in records:
        try:
            text = registry.raw_text(record)
        except FileNotFoundError:
            continue
        spans = _spans_for_source(record.source_id, text, tokens)
        out.extend(spans[:_MAX_SPANS_PER_SOURCE])
        if len(out) >= _MAX_SPANS_TOTAL:
            break
    return out[:_MAX_SPANS_TOTAL]


def _spans_for_source(
    source_id: str,
    text: str,
    tokens: list[str],
) -> list[VerbatimSpan]:
    """Return non-overlapping bounded snippets around token matches.

    Each match becomes a snippet ``[start, end)`` where ``start`` and
    ``end`` are character offsets into ``text`` snapped outward to the
    nearest whitespace. Overlapping or duplicate snippets are collapsed
    by ``(char_start, char_end)``.
    """
    lowered = text.lower()
    seen_ranges: set[tuple[int, int]] = set()
    spans: list[VerbatimSpan] = []
    for token in tokens:
        if not token:
            continue
        search_from = 0
        while True:
            pos = lowered.find(token, search_from)
            if pos == -1:
                break
            start, end = _snap_window(text, pos, pos + len(token))
            if (start, end) not in seen_ranges:
                seen_ranges.add((start, end))
                excerpt = text[start:end]
                spans.append(
                    VerbatimSpan(
                        source_id=source_id,
                        excerpt=excerpt,
                        excerpt_hash=_excerpt_hash(excerpt),
                        char_start=start,
                        char_end=end,
                    )
                )
            search_from = pos + max(1, len(token))
            if len(spans) >= _MAX_SPANS_PER_SOURCE:
                break
        if len(spans) >= _MAX_SPANS_PER_SOURCE:
            break
    return spans


def _snap_window(text: str, match_start: int, match_end: int) -> tuple[int, int]:
    """Expand ``[match_start, match_end)`` by ±_SNIPPET_RADIUS, snapped
    to whitespace. Never extends past the text bounds.
    """
    n = len(text)
    raw_start = max(0, match_start - _SNIPPET_RADIUS)
    raw_end = min(n, match_end + _SNIPPET_RADIUS)

    # Snap left edge to a whitespace boundary so we do not split a word.
    if raw_start > 0:
        nl = text.rfind(" ", 0, raw_start)
        if nl == -1:
            nl = text.rfind("\n", 0, raw_start)
        if nl != -1:
            raw_start = nl + 1
    # Snap right edge similarly.
    if raw_end < n:
        nr = text.find(" ", raw_end)
        if nr == -1:
            nr = text.find("\n", raw_end)
        if nr != -1:
            raw_end = nr

    return raw_start, raw_end


def _excerpt_hash(excerpt: str) -> str:
    return "sha256:" + hashlib.sha256(excerpt.encode("utf-8")).hexdigest()


# ---- structure-item retrieval -----------------------------------------


def _retrieve_structure_items(
    *,
    workspace: Workspace,
    structure_hints: list[SearchHit] | None,
) -> list[StructureItemHit]:
    """Rehydrate sidecar ``structure_item`` hits from the on-disk
    report, not from SQLite.

    The sidecar is advisory: every emitted ``StructureItemHit`` is
    rebuilt from the YAML report under ``state/structure/``. A stale
    sidecar row pointing at a deleted report, a retracted source, a
    source whose class or content hash changed, or a missing symbol
    path simply drops out here — no partial data reaches
    ``QueryResult``.
    """
    if not structure_hints:
        return []
    registry = SourceRegistry(workspace)
    records = {rec.source_id: rec for rec in registry.iter_records()}
    out: list[StructureItemHit] = []
    seen: set[tuple[str, str]] = set()
    for hit in structure_hints:
        if hit.doc_type != "structure_item":
            continue
        source_id = hit.source_id
        symbol_path = hit.structure_symbol_path
        if not isinstance(source_id, str) or not isinstance(symbol_path, str):
            continue
        key = (source_id, symbol_path)
        if key in seen:
            continue
        record = records.get(source_id)
        if record is None or record.status == "retracted":
            continue
        report_path = workspace.structure_report_path(source_id)
        if not report_path.is_file():
            continue
        try:
            data = yaml.safe_load(report_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("version") != "structure_report_v1":
            continue
        report_source_class = data.get("source_class")
        report_language = data.get("language")
        report_content_hash = data.get("content_hash")
        items = data.get("items")
        if not (
            isinstance(report_source_class, str)
            and isinstance(report_language, str)
            and isinstance(report_content_hash, str)
            and isinstance(items, list)
        ):
            continue
        if record.source_class != report_source_class:
            continue
        if record.content_hash != report_content_hash:
            continue
        rehydrated = _find_structure_item(items, symbol_path=symbol_path, hit=hit)
        if rehydrated is None:
            continue
        kind, name, locator = rehydrated
        out.append(
            StructureItemHit(
                source_id=source_id,
                source_class=report_source_class,
                language=report_language,
                kind=kind,
                name=name,
                symbol_path=symbol_path,
                locator=locator,
                report_path=_rel_report_path(workspace, report_path),
            )
        )
        seen.add(key)
        if len(out) >= _MAX_STRUCTURE_ITEMS_TOTAL:
            break
    return out


def _find_structure_item(
    items: list[Any],
    *,
    symbol_path: str,
    hit: SearchHit,
) -> tuple[str, str, dict] | None:
    """Locate the report item that matches the sidecar hit.

    When the sidecar supplied a ``kind`` / ``name``, they are used as
    extra guards so a hit whose metadata no longer matches the
    canonical report is dropped.
    """
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("symbol_path") != symbol_path:
            continue
        kind = item.get("kind")
        name = item.get("name")
        locator = item.get("locator")
        if not (
            isinstance(kind, str)
            and isinstance(name, str)
            and isinstance(locator, dict)
        ):
            return None
        if hit.structure_kind is not None and hit.structure_kind != kind:
            return None
        if hit.structure_name is not None and hit.structure_name != name:
            return None
        return kind, name, dict(locator)
    return None


def _rel_report_path(workspace: Workspace, path) -> str:
    try:
        rel = path.resolve().relative_to(workspace.root.resolve())
    except ValueError:
        rel = path
    return rel.as_posix()


# ---- answer formatting --------------------------------------------------


def _format_answer(
    citations: list[dict],
    spans: list[VerbatimSpan],
    structure_items: list[StructureItemHit],
) -> str:
    """Deterministic textual rendering of the citations, spans, and
    structure-item hits.

    No model invocation. Each line cites either an authoritative claim,
    a verbatim source span, or a rehydrated structure item. The format
    is intentionally rigid so callers can parse / diff it across runs.
    """
    lines: list[str] = []
    if citations:
        lines.append(f"Retrieved {len(citations)} authoritative claim(s):")
        for c in citations:
            lines.append(
                f"- [{c['entity_id']}/{c['claim_id']}] {c['text']}"
            )
    if spans:
        if lines:
            lines.append("")
        lines.append(
            f"Retrieved {len(spans)} verbatim span(s) from index_only source(s):"
        )
        for s in spans:
            lines.append(
                f"- [{s.source_id} chars {s.char_start}:{s.char_end}] {s.excerpt}"
            )
    if structure_items:
        if lines:
            lines.append("")
        lines.append(
            f"Retrieved {len(structure_items)} structure item(s):"
        )
        for item in structure_items:
            lines.append(
                f"- [{item.source_id} {item.language} {item.kind}] {item.symbol_path}"
            )
    if not lines:
        lines.append(
            "No authoritative claims, index_only spans, or structure items matched the question."
        )
    return "\n".join(lines)


# ---- shared helpers -----------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def _score(tokens: list[str], text: str) -> int:
    lowered = text.lower()
    return sum(1 for t in tokens if t in lowered)
