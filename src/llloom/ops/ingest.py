"""`ingest` operation.

Typed ingest is the core workflow. See
``04_specification/operations_and_cli.md`` Â§ingest for the policy contract.

For ``claim_extract`` and ``claim_extract_and_view_render`` policies,
``ingest`` invokes :class:`LLMInvoke` once with the source text plus
the schema documents, parses the harness's structured output into
candidate claims via :func:`llloom.llm.output.parse_claim_extraction_output`,
verifies every candidate's evidence against the raw source, and
persists only the verified candidates. The pipeline is **batch
atomic**: any parse error or verifier failure refuses the whole
batch and persists nothing.

Two candidate input paths are supported and feed the same verifier:

- model output from :class:`LLMInvoke` (the production path; with the
  default :class:`NullModel` no candidates materialize because output
  is empty), and
- explicit ``seed_claims`` supplied by callers and tests. Seed claims
  are the deterministic mechanism for proving the pipeline without
  wiring a live model.

Both paths route through :class:`LLMInvoke` so the invocation log
records exactly what the harness saw. Neither path bypasses the
provenance verifier; persisted assertions always carry a mandatory
``excerpt_hash`` computed from the resolved source span.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Strip common failed-OCR placeholder markers before judging emptiness.
_OCR_PLACEHOLDER_RE = re.compile(
    r"\*\*\s*==>\s*picture\s*\[[^\]]*\]\s*intentionally\s+omitted\s*<==\s*\*\*",
    re.IGNORECASE,
)


def _is_effectively_empty(text: str) -> bool:
    stripped = _OCR_PLACEHOLDER_RE.sub("", text).strip()
    return not stripped

from llloom.claims.models import (
    CLAIM_STATUSES,
    Assertion,
    EntityContainer,
    Evidence,
    Locator,
    MergeProposal,
    RenderTarget,
)
from llloom.claims.store import ClaimStore
from llloom.claims.verifier import (
    VerifierMismatch,
    compute_excerpt_hash,
    preview_excerpt,
    resolve_excerpt,
    verify_assertion,
)
from llloom.llm.harness import (
    LLMInvoke,
    SchemaDocument,
    SourceDocument,
    StructureItemContext,
)
from llloom.llm.output import (
    ModelOutputError,
    RawCandidate,
    parse_claim_extraction_output,
)
from llloom.ops._context import iso_now, operation, relative_posix
from llloom.ops.results import CreatedClaim, IngestResult
from llloom.pages.render import render_page_file_from_contributors, resolve_page_path
from llloom.schema.policy import Schema, load_schema
from llloom.sources.registry import SourceRegistry, SourceRegistryError
from llloom.state.fingerprints import FingerprintStore
from llloom.state.journal import OperationJournal
from llloom.structured import (
    StructureExtractError,
    extract_structure,
    write_structure_report,
)
from llloom.workspace.layout import Workspace


@dataclass
class SeedClaim:
    """Deterministic candidate claim supplied by a caller or test.

    The locator is resolved against the current raw source to derive
    ``excerpt_hash`` (if not supplied) and the optional verbatim excerpt.

    ``status`` is optional: a per-claim explicit lifecycle state wins
    when supplied (e.g. ``status="validated"``), otherwise the
    operation-level ``ingest(..., seed_claim_status=...)`` default
    applies. Slice 070 changed the default from the string ``"draft"``
    to ``None`` so the two signals can be distinguished. Model-backed
    candidates from :func:`_seed_from_raw` always carry an explicit
    status string (the parser fills it in) and are therefore
    unaffected by ``seed_claim_status``.
    """

    entity_id: str
    entity_type: str
    display_name: str
    claim_id: str
    claim_kind: str
    claim_text: str
    locator: Locator
    render_target: tuple[str, str] | None = None  # (page_id, block_id)
    excerpt_hash: str | None = None
    status: str | None = None


def ingest(
    workspace: Workspace,
    source_path: Path | str,
    *,
    source_id: str | None = None,
    source_class: str | None = None,
    seed_claims: list[SeedClaim] | None = None,
    seed_claim_status: str = "draft",
    llm: LLMInvoke | None = None,
    no_render: bool = False,
    structure_source_ids: list[str] | None = None,
) -> IngestResult:
    """Typed ingest.

    Parameters
    ----------
    workspace:
        Loaded workspace.
    source_path:
        Path to the raw source file (must already live under ``raw/``).
    source_id:
        Optional explicit source id. Derived from the filename otherwise.
    source_class:
        Optional override. If absent, the file extension + schema map
        must identify the class.
    seed_claims:
        Optional deterministic candidate claims. Tests use this to prove
        the end-to-end pipeline without a live model.
    seed_claim_status:
        Operation-level default lifecycle status for deterministic seed
        claims that do not carry a per-claim ``status`` override. Must
        be a value in :data:`llloom.claims.models.CLAIM_STATUSES`;
        anything else refuses the batch atomically before any
        persistence. Defaults to ``"draft"`` (the historical seed
        behavior). Has no effect on model-backed candidates from
        ``LLMInvoke`` — those always carry an explicit status from
        :func:`llloom.llm.output.parse_claim_extraction_output`.
        Curated seed scripts that intentionally land at ``"reviewed"``
        can opt in without needing the seed-manifest CLI (Slice 075)
        to ship first.
    llm:
        Optional LLMInvoke; a default with :class:`NullModel` is used if
        absent.
    no_render:
        Suppress the render step. Only meaningful for
        ``claim_extract_and_view_render``: verified claims still
        persist, but ``_render_targets`` is skipped and
        ``IngestResult.render_skipped`` is set ``True``. For every
        other policy (including ``claim_extract``, ``index_only``,
        ``structure_extract``, ``deny``) the flag has no effect.
    structure_source_ids:
        Optional explicit list of registered structure-source ids whose
        on-disk structure reports should be loaded and passed into
        :class:`LLMInvoke` as metadata-only
        :class:`~llloom.llm.harness.StructureItemContext` blocks. Only
        ``claim_extract`` and ``claim_extract_and_view_render`` consume
        them; ``index_only``, ``structure_extract``, and ``deny``
        return before any harness call and ignore the parameter. The
        narrative source being ingested is still the only place
        persisted claims may ground; structure context never widens
        what the verifier accepts. A requested structure source whose
        report is missing, stale, malformed, or whose registry record
        is retracted or wrong-class causes a clean batch-atomic
        refusal — silent omission would mislead the caller who
        explicitly requested the context.
    """
    schema = load_schema(workspace)
    harness = llm or LLMInvoke()
    registry = SourceRegistry(workspace)
    store = ClaimStore(workspace)
    journal = OperationJournal(workspace)

    src_path = Path(source_path).resolve()
    effective_id = source_id or SourceRegistry.derive_source_id(src_path)
    SourceRegistry.validate_source_id(effective_id)

    # Pick source class. Callers may pass one explicitly; otherwise infer
    # from extension. The first slice only handles markdown files unless
    # an override is provided.
    effective_class = source_class or _infer_source_class(src_path, schema)
    if effective_class not in schema.source_classes:
        raise ValueError(
            f"unknown source class {effective_class!r}; "
            f"known: {sorted(schema.source_classes)}"
        )

    policy = schema.resolve_ingest_policy(effective_class)

    result = IngestResult(
        source_id=effective_id,
        source_class=effective_class,
        policy=policy,
        registration_state="pending",
    )

    if seed_claim_status not in CLAIM_STATUSES:
        op_id = journal.new_op_id("ingest_refused")
        entry = journal.start(
            op_id=op_id,
            op_kind="ingest_refused",
            lock_id="(none)",
            planned_writes=[],
        )
        note = (
            f"invalid seed_claim_status {seed_claim_status!r}; "
            f"allowed lifecycle states: {sorted(CLAIM_STATUSES)}"
        )
        journal.refuse(entry, note)
        result.registration_state = "refused"
        result.refusal_reason = note
        result.extraction_errors.append(note)
        result.op_id = op_id
        return result

    # DENY is journaled but never acquires the lock (no mutation occurs).
    if policy == "deny":
        op_id = journal.new_op_id("ingest_denied")
        entry = journal.start(
            op_id=op_id,
            op_kind="ingest_denied",
            lock_id="(none)",
            planned_writes=[],
        )
        journal.refuse(
            entry, f"policy deny: source {effective_id} class {effective_class}"
        )
        result.registration_state = "refused"
        result.refusal_reason = "policy deny"
        result.op_id = op_id
        return result

    # Detect empty-source negative case (including failed-OCR
    # placeholder-only files such as ``NCCP Act of 1991.md``).
    raw_text = src_path.read_text(encoding="utf-8")
    if _is_effectively_empty(raw_text):
        op_id = journal.new_op_id("ingest_empty")
        entry = journal.start(
            op_id=op_id,
            op_kind="ingest_empty",
            lock_id="(none)",
            planned_writes=[],
        )
        journal.refuse(
            entry,
            f"empty-or-failed-OCR source {effective_id}: "
            f"no extractable text at {relative_posix(workspace, src_path)}",
        )
        result.registration_state = "refused"
        result.refusal_reason = "empty source"
        result.op_id = op_id
        return result

    with operation(
        workspace,
        op_kind="ingest",
        owner_id=f"ingest.{effective_id}",
    ) as ctx:
        record, reg_state = registry.register(
            source_id=effective_id,
            raw_path=src_path,
            source_class=effective_class,
        )
        result.registration_state = reg_state
        ctx.entry.touched_files.append(
            relative_posix(workspace, registry.record_path(effective_id))
        )

        # POLICY CUTOFF â€” must happen BEFORE any LLMInvoke construction.
        # Strict `index_only` and `structure_extract` both forbid the raw
        # source body from entering the harness as a SourceDocument or
        # any other typed input. Returning here (for `index_only`) or
        # taking the deterministic non-LLM `structure_extract` branch
        # guarantees that property by construction.
        if policy == "index_only":
            return _finalize(result, ctx, workspace)

        if policy == "structure_extract":
            source_class_obj = schema.source_class(effective_class)
            locator_type = source_class_obj.locator
            try:
                report = extract_structure(
                    raw_text,
                    source_id=effective_id,
                    source_class=effective_class,
                    locator_type=locator_type,
                    raw_path=relative_posix(workspace, src_path),
                    content_hash=record.content_hash,
                )
                report_path = write_structure_report(workspace, report)
            except StructureExtractError as exc:
                note = f"structure_extract failed: {exc}"
                result.refusal_reason = note
                ctx.entry.notes.append(note)
                return _finalize(result, ctx, workspace)
            rel_report = relative_posix(workspace, report_path)
            result.structure_reports.append(rel_report)
            ctx.entry.touched_files.append(rel_report)
            return _finalize(result, ctx, workspace)

        # claim_extract / claim_extract_and_view_render only beyond this
        # point. Build the typed LLMInvoke workspace for audit + canary
        # enforcement and persist the invocation log into the journal.
        source_class_obj = schema.source_class(effective_class)
        class_locator_type = source_class_obj.locator
        code_backed = class_locator_type == "code_v1"
        # NB: an earlier slice refused code-backed
        # `claim_extract_and_view_render` here as a staging choice. With
        # the code-evidence contract (declaration spans + attached
        # explanation spans) now fully in place, rendering is treated
        # as a post-persistence step: code-backed candidates flow
        # through the existing parser, the combined `code_v1`
        # validator, the verifier, and the same atomic persistence
        # pipeline as narrative claims; only after a successful batch
        # does `_render_targets(...)` run. The render path reads
        # authoritative claim state, not raw source text.
        allowed_locator_types: frozenset[str] = frozenset({class_locator_type})
        schema_docs = _load_schema_documents(workspace)
        try:
            structure_context = _load_structure_context(
                workspace, registry, structure_source_ids or []
            )
        except StructureContextError as exc:
            note = f"structure context error: {exc}"
            result.extraction_errors.append(note)
            result.refusal_reason = note
            ctx.entry.notes.append(note)
            return _finalize(result, ctx, workspace)
        output_text, invocation_log = harness.invoke(
            op_id=ctx.op_id,
            operation_kind="ingest",
            source_documents=[
                SourceDocument(
                    source_id=effective_id,
                    source_class=effective_class,
                    text=raw_text,
                )
            ],
            schema_documents=schema_docs,
            structure_items=structure_context,
        )
        # Record an invocation summary on the journal entry. The summary
        # contains class + content hash per typed input, never the source
        # text itself.
        ctx.entry.invocation_logs.append(invocation_log.to_mapping())

        # Parse model output into candidate claims. Empty output yields
        # zero candidates (the NullModel path). Malformed output is a
        # batch-atomic refusal: nothing persists.
        try:
            model_candidates = _model_candidates_from_output(
                output_text, allowed_locator_types=allowed_locator_types
            )
        except ModelOutputError as exc:
            note = f"model output parse error: {exc}"
            result.extraction_errors.append(note)
            result.refusal_reason = f"extraction failed: {note}"
            ctx.entry.notes.append(note)
            return _finalize(result, ctx, workspace)

        # Code-backed claim_extract: every emitted code_v1 locator must
        # match either a declaration-level structure item OR an attached
        # explanation span (leading line-comment block above a
        # declaration, or a Python docstring on the line immediately
        # below a class / function / async-function declaration), from
        # a transient walk of the current raw source. Detached
        # comments and arbitrary body spans are refused here.
        if code_backed and model_candidates:
            try:
                _validate_code_v1_claim_locators(
                    candidates=model_candidates,
                    source_text=raw_text,
                    source_id=effective_id,
                    source_class=effective_class,
                    raw_path=relative_posix(workspace, src_path),
                    content_hash=record.content_hash,
                )
            except CodeClaimContractError as exc:
                note = f"code_v1 claim validation failed: {exc}"
                result.extraction_errors.append(note)
                result.refusal_reason = f"extraction failed: {note}"
                ctx.entry.notes.append(note)
                return _finalize(result, ctx, workspace)

        all_candidates: list[SeedClaim] = list(model_candidates) + list(
            seed_claims or []
        )

        if all_candidates:
            try:
                claims_out, entities_out, merge_props = _apply_candidates(
                    workspace=workspace,
                    store=store,
                    source_id=effective_id,
                    candidates=all_candidates,
                    source_text=raw_text,
                    seed_claim_status=seed_claim_status,
                )
            except ClaimVerificationError as exc:
                # Batch atomic: nothing persists if any candidate fails.
                # Phase 1 raises before any write, so no rollback needed.
                note = str(exc)
                result.extraction_errors.append(note)
                result.refusal_reason = f"extraction failed: {note}"
                ctx.entry.notes.append(note)
                return _finalize(result, ctx, workspace)
            result.claims_created.extend(claims_out)
            result.entities_touched.extend(entities_out)
            result.merge_proposals_created.extend(merge_props)
            ctx.entry.touched_files.extend(
                f"claims/entities/{eid}.yaml" for eid in entities_out
            )

        if policy == "claim_extract_and_view_render":
            if no_render:
                # Caller asked to suppress rendering. Verified claims are
                # already persisted above; the page render step is the
                # only thing we skip. The next ``llloom render`` (or any
                # later ``ingest`` without ``--no-render``) will catch
                # the affected pages up.
                result.render_skipped = True
                ctx.entry.notes.append(
                    "render skipped at caller request (no_render=True)"
                )
            else:
                pages = _render_targets(workspace, store, result.entities_touched)
                result.pages_rendered.extend(pages)
                ctx.entry.touched_files.extend(pages)

        return _finalize(result, ctx, workspace)


# ---- helpers ------------------------------------------------------------


def _finalize(
    result: IngestResult,
    ctx,
    workspace: Workspace,
) -> IngestResult:
    result.op_id = ctx.op_id
    return result


def _infer_source_class(path: Path, schema: Schema) -> str:
    """Infer source class from extension.

    First slice: ``.md`` => first markdown-family class registered in
    the schema. If multiple, prefer ``markdown_prose``; else the first.
    """
    ext = path.suffix.lower()
    if ext != ".md":
        raise ValueError(
            f"first slice supports markdown (.md) only; got {ext!r} for {path}"
        )
    if "markdown_prose" in schema.source_classes:
        return "markdown_prose"
    for name, sc in schema.source_classes.items():
        if sc.locator in ("markdown_prose_v1", "legal_act_v1"):
            return name
    raise ValueError("no markdown source class registered in schema")


class StructureContextError(Exception):
    """Raised when a caller-requested structure source cannot be loaded
    as deterministic metadata-only context for ``claim_extract``.

    The caller explicitly named the structure source, so silent
    omission would mislead them. Surfaces as a batch-atomic refusal at
    the ``ingest`` boundary; no harness call, no claim persistence, no
    page render.
    """


def _load_structure_context(
    workspace: Workspace,
    registry: SourceRegistry,
    structure_source_ids: list[str],
) -> list[StructureItemContext]:
    """Rehydrate metadata-only structure context from on-disk reports.

    For every requested ``source_id``:

    - the registry record must exist, not be retracted, and match the
      report's ``source_class`` / ``content_hash``;
    - the report file at ``state/structure/<source_id>.yaml`` must
      exist, parse as YAML, carry ``version == structure_report_v1``,
      and agree on the same ``source_id``;
    - every item in the report becomes one ``StructureItemContext``
      with structure metadata only (no scalar values, no comments, no
      code bodies, no locator excerpt text).

    Any failure raises :class:`StructureContextError`. Order of items
    is the report's own deterministic order; cross-source order
    follows the caller's input order.
    """
    items: list[StructureItemContext] = []
    seen: set[str] = set()
    for sid in structure_source_ids:
        if sid in seen:
            continue
        seen.add(sid)
        try:
            record = registry.load(sid)
        except SourceRegistryError as exc:
            raise StructureContextError(
                f"structure source {sid!r} is not registered: {exc}"
            ) from exc
        if record.status != "registered":
            raise StructureContextError(
                f"structure source {sid!r} is {record.status}, not registered"
            )
        report_path = workspace.structure_report_path(sid)
        if not report_path.is_file():
            raise StructureContextError(
                f"structure source {sid!r} has no report at "
                f"state/structure/{sid}.yaml; re-ingest with "
                f"structure_extract"
            )
        try:
            loaded = yaml.safe_load(report_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise StructureContextError(
                f"structure source {sid!r}: report YAML parse failed: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise StructureContextError(
                f"structure source {sid!r}: report root is not a mapping"
            )
        if loaded.get("version") != "structure_report_v1":
            raise StructureContextError(
                f"structure source {sid!r}: unsupported report version "
                f"{loaded.get('version')!r}"
            )
        if loaded.get("source_id") != sid:
            raise StructureContextError(
                f"structure source {sid!r}: report claims source_id "
                f"{loaded.get('source_id')!r}"
            )
        if loaded.get("source_class") != record.source_class:
            raise StructureContextError(
                f"structure source {sid!r}: report source_class "
                f"{loaded.get('source_class')!r} != registry "
                f"{record.source_class!r}"
            )
        if loaded.get("content_hash") != record.content_hash:
            raise StructureContextError(
                f"structure source {sid!r}: report content_hash is stale "
                f"({loaded.get('content_hash')!r} != registry "
                f"{record.content_hash!r}); re-ingest with structure_extract"
            )
        language = str(loaded.get("language") or "")
        report_rel = relative_posix(workspace, report_path)
        raw_items = loaded.get("items") or []
        if not isinstance(raw_items, list):
            raise StructureContextError(
                f"structure source {sid!r}: report items field is not a list"
            )
        for entry in raw_items:
            if not isinstance(entry, dict):
                raise StructureContextError(
                    f"structure source {sid!r}: malformed item entry"
                )
            items.append(
                StructureItemContext(
                    source_id=sid,
                    source_class=record.source_class,
                    language=language,
                    kind=str(entry.get("kind") or ""),
                    name=str(entry.get("name") or ""),
                    symbol_path=str(entry.get("symbol_path") or ""),
                    report_path=report_rel,
                )
            )
    return items


def _load_schema_documents(workspace: Workspace) -> list[SchemaDocument]:
    docs: list[SchemaDocument] = []
    for name in ("source_classes.yaml", "ingest_policies.yaml", "page_classes.yaml"):
        path = workspace.schema / name
        if path.is_file():
            docs.append(
                SchemaDocument(name=name, text=path.read_text(encoding="utf-8"))
            )
    return docs


class ClaimVerificationError(Exception):
    """Raised by Phase 1 of :func:`_apply_candidates` when any candidate
    fails locator resolution, hash check, or assertion verification.

    Caught by the ``ingest`` operation and turned into a batch-atomic
    refusal: no candidate from the batch is persisted.
    """


def _model_candidates_from_output(
    output_text: str,
    *,
    allowed_locator_types: frozenset[str] | None = None,
) -> list["SeedClaim"]:
    """Parse model output into the same ``SeedClaim`` shape the seed-path uses."""
    raw = parse_claim_extraction_output(
        output_text, allowed_locator_types=allowed_locator_types
    )
    return [_seed_from_raw(c) for c in raw]


class CodeClaimContractError(Exception):
    """Raised when a model-emitted ``code_v1`` locator does not match a
    deterministic admissible span for the current source.

    Two surfaces are admitted on the code-backed ``claim_extract`` path:

    1. **Declarations** (the first ``code_v1`` slice) — every locator
       must match a structure-item locator from a transient walk of the
       current raw source on the six addressable keys
       (``locator_type``, ``path``, ``start_line``, ``start_col``,
       ``end_line``, ``end_col``). Class / function / method / type /
       interface / trait / enum / struct definitions only.

    2. **Attached explanation** (this slice) — contiguous line-comment
       blocks immediately above a declaration, and Python triple-quoted
       docstrings on the line immediately below a class / function /
       async-function declaration. Both shapes are enumerated
       deterministically from the current raw source text; locators
       must cover whole lines and must attach to a structure item.

    Anything else — detached comments, free-form body spans, fabricated
    line ranges — refuses the whole batch.
    """


_CODE_V1_LOCATOR_KEYS: tuple[str, ...] = (
    "locator_type",
    "path",
    "start_line",
    "start_col",
    "end_line",
    "end_col",
)


def _validate_code_v1_declaration_locators(
    *,
    candidates: list["SeedClaim"],
    source_text: str,
    source_id: str,
    source_class: str,
    raw_path: str,
    content_hash: str,
) -> None:
    """Refuse code_v1 candidates that do not match a deterministic
    declaration span for the current source.

    Builds a transient ``StructureReport`` from the current raw source
    and compares each candidate's ``code_v1`` locator against the
    report's item locators on the six addressable keys
    (``locator_type``, ``path``, ``start_line``, ``start_col``,
    ``end_line``, ``end_col``). Symbol-path / kind / name from the
    structure item are not part of the comparison because the
    provider is not asked to echo them; alignment of the addressable
    span is the contract.
    """
    try:
        report = extract_structure(
            source_text,
            source_id=source_id,
            source_class=source_class,
            locator_type="code_v1",
            raw_path=raw_path,
            content_hash=content_hash,
        )
    except StructureExtractError as exc:
        raise CodeClaimContractError(
            f"could not build transient structure report for {source_id!r}: "
            f"{exc}"
        ) from exc
    allowed_spans: set[tuple] = {
        tuple(item.locator.get(k) for k in _CODE_V1_LOCATOR_KEYS)
        for item in report.items
    }
    for cand in candidates:
        if cand.locator.locator_type != "code_v1":
            continue
        emitted = (
            cand.locator.locator_type,
            cand.locator.path,
            cand.locator.start_line,
            cand.locator.start_col,
            cand.locator.end_line,
            cand.locator.end_col,
        )
        if emitted not in allowed_spans:
            raise CodeClaimContractError(
                f"candidate {cand.claim_id}: code_v1 locator does not match "
                "any declaration-level structure item from the current "
                "source (this slice admits only class / function / method / "
                "type / interface / trait / enum / struct declarations; "
                "comments, docstrings, and arbitrary body spans are "
                "deferred)"
            )


# ---- attached-explanation admission ------------------------------------


_LINE_COMMENT_PREFIXES: dict[str, str] = {
    ".py": "#",
    ".go": "//",
    ".rs": "//",
    ".ts": "//",
    # `.cs` admits both `//` line comments and `///` XML-doc comments
    # naturally: `///` lines pass `str.startswith("//")`, so the same
    # contiguous-comment-block rule above a declaration captures both
    # shapes without a second enumeration system.
    ".cs": "//",
}

# Languages whose declarations may carry a triple-quoted docstring as
# their first body statement. Extending this set beyond Python is a
# future slice — Go/Rust/TypeScript do not have the same convention and
# would smuggle arbitrary body strings into the explanation contract.
_DOCSTRING_LANGUAGES: frozenset[str] = frozenset({".py"})

_DOCSTRING_KINDS: frozenset[str] = frozenset(
    {"class", "function", "async_function"}
)


def _code_path_suffix(raw_path: str) -> str:
    idx = raw_path.rfind(".")
    if idx == -1 or idx == len(raw_path) - 1:
        return ""
    return raw_path[idx:].lower()


def _enumerate_attached_explanation_locators(
    *,
    source_text: str,
    raw_path: str,
    declarations,  # list[StructureItem]
) -> set[tuple]:
    """Return the set of allowed attached-explanation `code_v1` span
    tuples for the current source.

    Two shapes are admitted per declaration:

    - **leading comment block**: the contiguous lines immediately above
      ``decl.start_line`` whose stripped content begins with the
      language's line-comment prefix. The block must abut the
      declaration with no blank line in between. The emitted locator
      covers whole lines (``start_col == 1``, ``end_col`` = the last
      comment line's full length).

    - **Python docstring** (Python only): on the line immediately below
      ``decl.start_line`` (``decl.start_line + 1``), if the
      stripped content begins with ``\"\"\"`` or ``'''``, scan forward
      to the matching closing triple-quote. Both single-line and
      multi-line docstrings are admitted. The locator covers whole
      lines from the opening line through the closing line.

    The walk is deterministic and metadata-only: no scalar values,
    body text, or identifier text from the source enter the returned
    set. Tuples follow the six-key shape used by the declaration
    enumerator so the union check in the combined validator is
    homogeneous.
    """
    suffix = _code_path_suffix(raw_path)
    comment_prefix = _LINE_COMMENT_PREFIXES.get(suffix)
    is_docstring_lang = suffix in _DOCSTRING_LANGUAGES
    if comment_prefix is None and not is_docstring_lang:
        return set()
    lines = source_text.splitlines()  # drop terminators
    spans: set[tuple] = set()
    for decl in declarations:
        decl_start_line = int(decl.locator.get("start_line", 0))
        if decl_start_line < 1:
            continue
        if comment_prefix is not None:
            block = _collect_leading_comment_block(
                lines, decl_start_line=decl_start_line, prefix=comment_prefix
            )
            if block is not None:
                spans.add(_whole_line_span(raw_path, *block, lines=lines))
        if is_docstring_lang and decl.kind in _DOCSTRING_KINDS:
            block = _collect_python_docstring(
                lines, decl_start_line=decl_start_line
            )
            if block is not None:
                spans.add(_whole_line_span(raw_path, *block, lines=lines))
    return spans


def _whole_line_span(
    raw_path: str, start_line: int, end_line: int, *, lines: list[str]
) -> tuple:
    """Build a whole-line ``code_v1`` span tuple shaped like the
    declaration enumerator emits. ``end_col`` is the full character
    length of the last covered line; ``start_col`` is 1."""
    end_col = len(lines[end_line - 1]) if end_line <= len(lines) else 1
    if end_col < 1:
        end_col = 1
    return ("code_v1", raw_path, start_line, 1, end_line, end_col)


def _collect_leading_comment_block(
    lines: list[str], *, decl_start_line: int, prefix: str
) -> tuple[int, int] | None:
    """Walk upward from ``decl_start_line - 1`` collecting contiguous
    comment lines. Returns ``(start_line, end_line)`` 1-based, or
    ``None`` if no comment block immediately precedes the declaration.
    """
    cursor = decl_start_line - 1
    end_line: int | None = None
    while cursor >= 1:
        line = lines[cursor - 1] if cursor - 1 < len(lines) else ""
        stripped = line.lstrip()
        if stripped.startswith(prefix):
            end_line = cursor if end_line is None else end_line
            start_line = cursor
            cursor -= 1
            continue
        break
    if end_line is None:
        return None
    return (cursor + 1, end_line)


_DOCSTRING_QUOTES: tuple[str, ...] = ('"""', "'''")


def _collect_python_docstring(
    lines: list[str], *, decl_start_line: int
) -> tuple[int, int] | None:
    """Detect a triple-quoted docstring on ``decl_start_line + 1``.

    Returns ``(start_line, end_line)`` 1-based, or ``None`` if the
    line below the declaration is not the start of a docstring. Both
    single-line (``\"\"\"...\"\"\"``) and multi-line docstring shapes
    are admitted. Inline string-expression statements that happen to
    start with a triple quote inside arbitrary body text are excluded
    by construction because we only look at ``decl_start_line + 1``.
    """
    open_line = decl_start_line + 1
    if open_line > len(lines):
        return None
    raw = lines[open_line - 1]
    stripped = raw.lstrip()
    quote: str | None = None
    for q in _DOCSTRING_QUOTES:
        if stripped.startswith(q):
            quote = q
            break
    if quote is None:
        return None
    after_open = stripped[len(quote):]
    # Single-line docstring: closing quote on the same line.
    if quote in after_open:
        return (open_line, open_line)
    # Multi-line: scan forward for the closing triple-quote.
    cursor = open_line + 1
    while cursor <= len(lines):
        if quote in lines[cursor - 1]:
            return (open_line, cursor)
        cursor += 1
    return None


def _validate_code_v1_claim_locators(
    *,
    candidates: list["SeedClaim"],
    source_text: str,
    source_id: str,
    source_class: str,
    raw_path: str,
    content_hash: str,
) -> None:
    """Refuse `code_v1` candidates that do not match either a
    declaration-level structure item or an attached-explanation span.

    This is the combined admission point on the code-backed
    `claim_extract` path. The narrower
    `_validate_code_v1_declaration_locators(...)` remains callable as
    a standalone helper for tests and for callers that explicitly
    want the declaration-only contract; the ingest path uses this
    combined wrapper so both surfaces are checked in one transient
    structure walk.
    """
    try:
        report = extract_structure(
            source_text,
            source_id=source_id,
            source_class=source_class,
            locator_type="code_v1",
            raw_path=raw_path,
            content_hash=content_hash,
        )
    except StructureExtractError as exc:
        raise CodeClaimContractError(
            f"could not build transient structure report for {source_id!r}: "
            f"{exc}"
        ) from exc
    declaration_spans: set[tuple] = {
        tuple(item.locator.get(k) for k in _CODE_V1_LOCATOR_KEYS)
        for item in report.items
    }
    explanation_spans = _enumerate_attached_explanation_locators(
        source_text=source_text,
        raw_path=raw_path,
        declarations=report.items,
    )
    allowed_spans = declaration_spans | explanation_spans
    for cand in candidates:
        if cand.locator.locator_type != "code_v1":
            continue
        emitted = (
            cand.locator.locator_type,
            cand.locator.path,
            cand.locator.start_line,
            cand.locator.start_col,
            cand.locator.end_line,
            cand.locator.end_col,
        )
        if emitted not in allowed_spans:
            raise CodeClaimContractError(
                f"candidate {cand.claim_id}: code_v1 locator does not match "
                "any declaration-level structure item OR attached "
                "explanation span (leading line-comment block immediately "
                "above a declaration, or a Python triple-quoted docstring "
                "on the line immediately below a class / function / "
                "async-function declaration) from the current source; "
                "arbitrary body spans and detached comments are deferred"
            )


def _seed_from_raw(raw: RawCandidate) -> "SeedClaim":
    return SeedClaim(
        entity_id=raw.entity_id,
        entity_type=raw.entity_type,
        display_name=raw.display_name,
        claim_id=raw.claim_id,
        claim_kind=raw.claim_kind,
        claim_text=raw.claim_text,
        locator=raw.locator,
        render_target=raw.render_target,
        excerpt_hash=raw.excerpt_hash,
        status=raw.status,
    )


def _apply_candidates(
    *,
    workspace: Workspace,
    store: ClaimStore,
    source_id: str,
    candidates: list[SeedClaim],
    source_text: str,
    seed_claim_status: str,
    dry_run: bool = False,
) -> tuple[list[CreatedClaim], list[str], list[str]]:
    """Two-phase atomic application of candidate claims.

    Phase 1 — Validate every candidate. Resolve the locator, compute
    the canonical excerpt hash, build the assertion, and verify it.
    Also resolve the effective lifecycle status (per-candidate override
    wins, otherwise ``seed_claim_status``) and refuse the whole batch
    if the effective status is not in
    :data:`llloom.claims.models.CLAIM_STATUSES`. If any candidate
    fails, raise :class:`ClaimVerificationError` BEFORE any write
    occurs.

    Phase 2 — Persist every validated candidate. Each entity write is
    atomic via temp-file-and-rename. Alias conflicts queue merge
    proposals (write-as-new, queue-for-merge).

    The two phases together provide batch atomicity: either every
    candidate persists, or none does. The returned ``CreatedClaim``
    items reflect the persisted ``Assertion.status`` and
    ``Assertion.verification_status`` so the ingest result can
    surface authority at the moment of creation.

    Slice 075 added the ``dry_run`` keyword so deterministic seed
    manifest application can reuse the same Phase 1 verifier walk
    without persisting anything. ``dry_run=True`` runs the full
    Phase 1 (still raises :class:`ClaimVerificationError` on any
    failure) and then returns
    ``([], sorted({c.entity_id for c in candidates}), [])`` instead
    of running Phase 2. Existing callers (`ingest(...)`) use the
    default ``False`` and are byte-identical.
    """
    now = iso_now()

    @dataclass
    class _Validated:
        candidate: "SeedClaim"
        assertion: Assertion
        existing_owner: str | None

    validated: list[_Validated] = []

    # Phase 1: validate all.
    for cand in candidates:
        effective_status = (
            cand.status if cand.status is not None else seed_claim_status
        )
        if effective_status not in CLAIM_STATUSES:
            raise ClaimVerificationError(
                f"candidate {cand.claim_id}: invalid lifecycle status "
                f"{effective_status!r}; allowed: {sorted(CLAIM_STATUSES)}"
            )

        try:
            excerpt = resolve_excerpt(
                Evidence(
                    source_id=source_id,
                    locator=cand.locator,
                    excerpt_hash="placeholder",
                ),
                source_text,
            )
        except Exception as exc:  # SpanResolutionError or anything raised
            raise ClaimVerificationError(
                f"candidate {cand.claim_id}: locator did not resolve: {exc}"
            ) from exc

        canonical_hash = compute_excerpt_hash(excerpt, cand.locator.locator_type)
        if cand.excerpt_hash and cand.excerpt_hash != canonical_hash:
            mismatch = VerifierMismatch(
                claim_id=cand.claim_id,
                source_id=source_id,
                locator_type=cand.locator.locator_type,
                stored_hash=cand.excerpt_hash,
                computed_hash=canonical_hash,
                current_preview=preview_excerpt(excerpt) or "",
                stored_preview=None,
            )
            raise ClaimVerificationError(
                f"candidate {cand.claim_id}: supplied excerpt_hash does not "
                f"match computed hash. {mismatch}"
            )

        render_targets: list[RenderTarget] = []
        if cand.render_target is not None:
            page_id, block_id = cand.render_target
            render_targets.append(RenderTarget(page_id=page_id, block_id=block_id))

        assertion = Assertion(
            claim_id=cand.claim_id,
            subject_id=cand.entity_id,
            claim_kind=cand.claim_kind,
            claim_text=cand.claim_text,
            evidence=[
                Evidence(
                    source_id=source_id,
                    locator=cand.locator,
                    excerpt_hash=canonical_hash,
                    excerpt=excerpt,
                )
            ],
            render_targets=render_targets,
            status=effective_status,
            verification_status="unverified",
            created_at=now,
            updated_at=now,
        )

        verification = verify_assertion(assertion, {source_id: source_text})
        if not verification.passed:
            # Use the structured mismatch (when present) so the
            # ``IngestResult.extraction_errors`` line carries the
            # source id, locator type, both hashes, and bounded
            # preview that an agent or human needs to debug the
            # refusal. Falls back to the verifier's textual notes
            # for non-hash failures (e.g. span resolution).
            if verification.mismatches:
                detail = "; ".join(str(m) for m in verification.mismatches)
            else:
                detail = "; ".join(verification.notes)
            raise ClaimVerificationError(
                f"candidate {cand.claim_id} failed verification: {detail}"
            )
        assertion.verification_status = "verified"

        # Alias conflict check is read-only at this stage.
        existing_owner = store.find_entity_by_alias(cand.display_name)
        validated.append(
            _Validated(candidate=cand, assertion=assertion, existing_owner=existing_owner)
        )

    # Slice 075: dry-run short-circuit. Phase 1 succeeded above for
    # every candidate; the seed-manifest path only needs the
    # verifier walk's success/failure verdict and the planned
    # entity-id set. Return the same shape as the persistence path
    # with empty written records.
    if dry_run:
        planned_entities = sorted({v.candidate.entity_id for v in validated})
        return [], planned_entities, []

    # Phase 2: persist all.
    created: list[CreatedClaim] = []
    entity_ids: set[str] = set()
    proposal_ids: list[str] = []
    for v in validated:
        cand = v.candidate
        if v.existing_owner is not None and v.existing_owner != cand.entity_id:
            proposal_id = f"mp.{cand.entity_id}.into.{v.existing_owner}"
            proposal = MergeProposal(
                proposal_id=proposal_id,
                source_entity_id=cand.entity_id,
                target_entity_id=v.existing_owner,
                proposed_alias_text=cand.display_name,
                reason=(
                    f"alias text {cand.display_name!r} already owned by "
                    f"{v.existing_owner}; write-as-new, queue-for-merge"
                ),
                created_at=now,
            )
            store.save_proposal(proposal)
            proposal_ids.append(proposal_id)

        store.upsert_assertion(
            entity_id=cand.entity_id,
            entity_type=cand.entity_type,
            display_name=cand.display_name,
            assertion=v.assertion,
        )
        created.append(
            CreatedClaim(
                claim_id=cand.claim_id,
                entity_id=cand.entity_id,
                status=v.assertion.status,
                verification_status=v.assertion.verification_status,
            )
        )
        entity_ids.add(cand.entity_id)

    return created, sorted(entity_ids), proposal_ids


def _render_targets(
    workspace: Workspace,
    store: ClaimStore,
    entity_ids: list[str],
) -> list[str]:
    """Render every page targeted by claims on the given entities.

    Slice 071: page/block-centric. The set of pages to render is the
    union of every touched entity's ``render_targets.page_id``s; each
    such page is rendered exactly once with the **full** contributor
    set from the canonical store (not only the touched entities). This
    closes the WME Audio field-feedback bug: previously the loop
    rendered one entity at a time and the later writer overwrote the
    earlier contributor's claim block.
    """
    fingerprints = FingerprintStore(workspace)

    affected_pages: set[str] = set()
    for eid in entity_ids:
        if not store.exists(eid):
            continue
        entity = store.load_entity(eid)
        for assertion in entity.assertions:
            for target in assertion.render_targets:
                affected_pages.add(target.page_id)

    rendered: list[str] = []
    for page_id in sorted(affected_pages):
        page_path = resolve_page_path(workspace, page_id)
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
        result = render_page_file_from_contributors(
            workspace, page_path, contributors
        )
        fingerprints.set(result.page_id, result.fingerprint)
        rendered.append(relative_posix(workspace, page_path))
    return rendered

