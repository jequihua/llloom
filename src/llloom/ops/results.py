"""Structured result objects for every user-facing operation.

Results are dataclasses so CLI, tests, and later MCP wrappers can all
consume the same shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llloom.claims.verifier import VerifierMismatch
from llloom.state.seed_reports import SeedExcerptCheck


@dataclass(frozen=True)
class CreatedClaim:
    """Persisted claim summary returned by ``ingest`` for every claim
    written in a batch.

    Slice 070 introduced this dataclass so the create result no longer
    looks more authoritative than it is: agents can see at the moment
    of creation whether a claim landed at ``draft`` (the deterministic
    default for curated seeds) or at a higher lifecycle state that the
    caller requested through ``ingest(..., seed_claim_status=...)``.
    ``verification_status`` reflects the verifier's post-resolution
    decision and is ``"verified"`` for every persisted record (any
    verifier failure refuses the whole batch before persistence).
    """

    claim_id: str
    entity_id: str
    status: str
    verification_status: str


@dataclass
class IngestResult:
    source_id: str
    source_class: str
    policy: str
    registration_state: str  # new | unchanged | refused
    claims_created: list[CreatedClaim] = field(default_factory=list)
    entities_touched: list[str] = field(default_factory=list)
    pages_rendered: list[str] = field(default_factory=list)
    merge_proposals_created: list[str] = field(default_factory=list)
    # Workspace-relative POSIX paths of derived structure reports
    # written by ``structure_extract``. Empty for every other policy.
    structure_reports: list[str] = field(default_factory=list)
    # Per-candidate notes from the model-output parser or the verifier.
    # Populated only on the model-backed extraction path. A non-empty
    # list with a None refusal_reason means non-fatal warnings (no such
    # cases in the first model slice; extraction is batch-atomic).
    extraction_errors: list[str] = field(default_factory=list)
    # True iff the policy resolved to ``claim_extract_and_view_render``
    # but the caller passed ``no_render=True`` (CLI ``--no-render``),
    # so the render step was deliberately suppressed. False on every
    # other path, including policies that never render.
    render_skipped: bool = False
    refusal_reason: str | None = None
    op_id: str = ""

    @property
    def succeeded(self) -> bool:
        return self.refusal_reason is None


@dataclass
class VerifyResult:
    verified: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Structured ``excerpt_hash`` mismatches surfaced from the
    # verifier. Populated alongside the textual entries in ``notes``
    # so agent / human review surfaces can render either form. Each
    # item carries the source id, locator type, both hashes, and
    # bounded previews of the current + stored span text. Previews
    # are deliberately bounded; this field is not a back-channel for
    # raw source bodies.
    mismatches: list[VerifierMismatch] = field(default_factory=list)
    passed: bool = True


@dataclass(frozen=True)
class RenderPlanContributor:
    """One contributing entity inside a :class:`RenderPlanEntry`.

    Slice 073 added this dataclass for the read-only render plan
    surface (``llloom render --dry-run`` / ``--list-targets``). The
    ``claim_ids`` are the render-visible assertion ids this entity
    contributes to the targeted block, sorted by ``claim_id`` for
    byte-stable plan output.
    """

    entity_id: str
    display_name: str
    claim_ids: list[str] = field(default_factory=list)


@dataclass
class RenderPlanEntry:
    """One page's read-only render plan entry.

    Slice 073 added this dataclass for the dry-run / list-targets
    surfaces. The plan reflects what the existing union renderer
    would do without acquiring the workspace lock, opening a render
    journal entry, or writing any page or fingerprint. Marker health
    surfaces page-parse problems instead of raising on the read-only
    path; real render still fails hard on the same conditions.
    """

    target: str  # "page:<page_id>"
    page_id: str
    page_path: str  # workspace-relative POSIX
    block_id: str | None = None
    contributors: list[RenderPlanContributor] = field(default_factory=list)
    contributing_claim_ids: list[str] = field(default_factory=list)
    marker_health: str = "ok"  # ok | missing_page | parse_error
    marker_message: str | None = None
    content_would_change: bool | None = None
    fingerprint_would_change: bool | None = None
    planned_fingerprint: str | None = None
    stored_fingerprint: str | None = None


@dataclass(frozen=True)
class PlannedSeedClaim:
    """One seed-manifest claim previewed before any persistence.

    Slice 075 added this dataclass for the deterministic seed
    manifest dry-run path. The ``status`` is the **effective** status
    after manifest defaults → source defaults → per-claim → CLI
    ``--status`` merge. ``source_id`` and ``entity_id`` reflect the
    targeted records; ``claim_id`` is the candidate's id.
    """

    source_id: str
    claim_id: str
    entity_id: str
    status: str


@dataclass
class SeedManifestResult:
    """Result of ``apply_seed_manifest(...)`` / ``llloom seed apply``.

    Slice 075 introduced this result. ``manifest_path`` is the
    workspace-relative POSIX path of the manifest YAML when it lives
    inside the workspace, otherwise the absolute path the caller
    supplied. ``sources_planned`` carries every source id the
    manifest names (in manifest order). ``claims_planned`` is the
    full list of :class:`PlannedSeedClaim` records including their
    merged effective status — populated on dry-run and real apply
    alike. ``claims_created`` carries :class:`CreatedClaim` records
    only on the real-apply success path; dry-run leaves it empty.
    ``entities_touched`` / ``pages_rendered`` / ``render_skipped``
    mirror the existing ingest result fields. ``refusal_reason`` is
    set on parse / shape / status / locator / verifier failures.
    ``op_ids`` collects every ``operation(...)`` journal op id the
    apply created (empty on dry-run; one entry on a single-source
    apply; one entry per source on multi-source apply).
    """

    manifest_path: str
    dry_run: bool = False
    no_render: bool = False
    sources_planned: list[str] = field(default_factory=list)
    claims_planned: list[PlannedSeedClaim] = field(default_factory=list)
    claims_created: list["CreatedClaim"] = field(default_factory=list)
    entities_touched: list[str] = field(default_factory=list)
    pages_rendered: list[str] = field(default_factory=list)
    render_skipped: bool = False
    refusal_reason: str | None = None
    op_ids: list[str] = field(default_factory=list)
    # Slice 076: durable audit evidence. ``report_path`` is the
    # workspace-relative POSIX path of the ``state/reports/updates/
    # <op_id>.yaml`` artifact written on a successful real apply (None
    # on dry-run, refusal, or any path that wrote no journal entry).
    # ``excerpt_checks`` carries one :class:`SeedExcerptCheck` per
    # claim that opted into ``excerpt_equality`` (or one per claim on
    # the success path when the report writer recorded resolved
    # excerpts even without a check). ``counts_before`` /
    # ``counts_after`` are workspace-wide source / entity / claim /
    # page totals captured at the start and end of the apply so
    # reviewers can audit deltas without re-walking the workspace.
    report_path: str | None = None
    excerpt_checks: list[SeedExcerptCheck] = field(default_factory=list)
    counts_before: dict[str, int] = field(default_factory=dict)
    counts_after: dict[str, int] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.refusal_reason is None


@dataclass
class RenderResult:
    rendered_pages: list[str] = field(default_factory=list)
    unchanged_pages: list[str] = field(default_factory=list)
    fingerprints: dict[str, str] = field(default_factory=dict)
    # Slice 073: dry-run / list-targets fields. ``plan`` is empty on
    # the mutating render path; populated only when ``dry_run`` or
    # ``list_targets`` is True. Both flags default to False so the
    # existing CLI / library surface is byte-identical.
    plan: list[RenderPlanEntry] = field(default_factory=list)
    dry_run: bool = False
    list_targets: bool = False


@dataclass
class LintResult:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    canary_hits: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures and not self.canary_hits


@dataclass
class ReconcileResult:
    actions: list[str] = field(default_factory=list)
    lock_cleared: bool = False
    journals_marked_interrupted: list[str] = field(default_factory=list)
    temp_files_removed: list[str] = field(default_factory=list)
    pages_rerendered: list[str] = field(default_factory=list)


@dataclass
class UnlockRecord:
    """Result of an ``unlock`` invocation.

    Two modes:

    - ``mode == "unlock_window"`` — bare ``llloom unlock <target>
      --reason "..."`` records a time-bounded maintenance window in
      the operation journal. ``lock_cleared`` is always ``False``;
      the workspace lock file is never touched. ``unlocked_at`` /
      ``expires_at`` describe the window.
    - ``mode == "clear_stale_lock"`` — ``llloom unlock
      --clear-stale --reason "..."`` clears the workspace lock only
      when ``WorkspaceLock.is_stale_recoverable(..., journal=...)``
      returns ``(True, ...)``. On success ``lock_cleared`` is
      ``True`` and the ``prior_*`` fields capture the cleared lock's
      identity. On refusal ``lock_cleared`` is ``False``,
      ``refused`` is ``True`` and ``refusal_reason`` explains the
      refusal in operator-actionable terms.
    """

    target: str
    reason: str
    unlocked_at: str
    expires_at: str
    op_id: str
    mode: str = "unlock_window"
    lock_cleared: bool = False
    refused: bool = False
    refusal_reason: str | None = None
    prior_op_id: str | None = None
    prior_owner_id: str | None = None
    prior_acquired_at: str | None = None
    prior_heartbeat_at: str | None = None
    # Slice 086: additive fields populated on the new
    # ``mode="clear_dead_owner_lock"`` path. Echo the optional
    # ``Lock.owner_pid`` / ``Lock.owner_hostname`` fields of the lock
    # that was cleared, plus the local owner-process classification
    # result (always ``"dead"`` on the success path; mirrors
    # :func:`llloom.state.lock.local_owner_pid_state` for refusals).
    # ``None`` on every other mode and every other refusal path so
    # the bare unlock-window and clear-stale shapes are unchanged.
    prior_owner_pid: int | None = None
    prior_owner_hostname: str | None = None
    prior_owner_pid_state: str | None = None


@dataclass
class SupersedeResult:
    """Result of ``supersede(workspace, *, old, by)`` / ``llloom supersede``.

    Slice 078 introduced this dataclass. ``old_target`` and
    ``new_target`` are the resolved fully-qualified
    ``claim:<entity_id>:<claim_id>`` strings so callers can paste
    them straight back into ``promote`` or ``claim-card``.
    ``old_from_status`` records the old claim's status before the
    transition; ``old_to_status`` is the constant ``"superseded"``
    on the success path. ``new_status`` mirrors the new claim's
    status (operator policy requires ``validated`` before a claim
    may supersede another). ``supersedes`` is the value written to
    ``new.assertion.supersedes`` — see the slice spec for the
    encoding choice. ``op_id`` is the ``op_kind="supersede"``
    journal id. On any pre-mutation refusal, ``refused`` is
    ``True``, ``reason`` is non-empty, and every other field
    carries a best-effort echo of the requested targets.
    """

    old_target: str
    new_target: str
    old_from_status: str
    old_to_status: str
    new_status: str
    supersedes: str
    op_id: str = ""
    refused: bool = False
    reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return not self.refused


@dataclass
class PromoteResult:
    target: str
    from_status: str
    to_status: str
    op_id: str
    refused: bool = False
    reason: str | None = None


@dataclass
class PageCreateResult:
    """Result of ``create_page(workspace, page_id=..., ...)`` /
    ``llloom page create ...`` (Slice 084).

    The operation writes one valid variant-(B) page stub under
    ``pages/<class_dir>/<tail>.md``. ``page_path`` is the
    workspace-relative POSIX path of the created stub. The marker
    ids are deterministic functions of the normalized ``page_id``
    (slashes replaced with ``.``, prefixed with ``claim_block.`` /
    ``commentary.``). ``status`` mirrors the YAML frontmatter
    ``status`` field of the stub and is the constant ``"draft"`` on
    the success path. On refusal, ``refusal_reason`` carries the
    explanation; pre-operation refusals raise :class:`PageCreateError`
    instead and never open a journal entry. ``op_id`` is the
    ``op_kind="page_create"`` journal entry id when the operation
    context opened (success or in-context refusal); otherwise empty.
    """

    page_id: str
    page_class: str
    page_path: str
    claim_block_id: str
    commentary_id: str
    status: str = "draft"
    op_id: str = ""
    refusal_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.refusal_reason is None


@dataclass
class PdfPrepArtifact:
    """One artifact produced by a PDF-prep bundle."""

    path: str  # workspace-relative POSIX
    kind: str  # e.g. "docling_markdown", "docling_json"
    sha256: str  # "sha256:<hex>"


@dataclass
class PdfPrepResult:
    """Result of ``prepare_pdf``.

    `succeeded` is true iff `status == "succeeded"`. On refusal /
    failure, `selected_artifact` is None and `refusal_reason` carries
    the explanation surfaced to the CLI.
    """

    prep_id: str
    status: str  # "succeeded" | "failed" | "refused"
    bundle_dir: str  # workspace-relative POSIX
    manifest_path: str  # workspace-relative POSIX
    source_pdf: str  # workspace-relative POSIX of the input PDF
    source_pdf_sha256: str
    provider: str = ""
    artifacts: list[PdfPrepArtifact] = field(default_factory=list)
    components: dict[str, str] = field(default_factory=dict)
    selected_artifact: PdfPrepArtifact | None = None
    refusal_reason: str | None = None
    op_id: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded" and self.refusal_reason is None


@dataclass
class RetractResult:
    source_id: str
    tombstone_path: str
    affected_claim_ids: list[str] = field(default_factory=list)
    affected_pages: list[str] = field(default_factory=list)
    rerendered_pages: list[str] = field(default_factory=list)
    op_id: str = ""


@dataclass
class VerbatimSpan:
    """Deterministic exact-retrieval result from an `index_only` source.

    Spans are produced by ``query`` against raw source text, never by an
    LLM. ``excerpt_hash`` is the SHA-256 of ``excerpt`` (matching the
    ``llloom.llm.harness.SourceSpan`` content-hash convention) so callers
    can correlate a returned span with the harness-side typed input class
    without re-hashing.
    """

    source_id: str
    excerpt: str
    excerpt_hash: str
    char_start: int
    char_end: int


@dataclass
class StructureItemHit:
    """Rehydrated structure-report item returned alongside a
    ``QueryResult``.

    Produced by ``query`` only when the optional search sidecar
    indexed a ``structure_item`` row whose report can be
    rehydrated from ``state/structure/<source_id>.yaml``. Carries
    structure metadata and a ``code_v1`` locator only — never
    scalar values, comments, docstrings, or code bodies.
    """

    source_id: str
    source_class: str
    language: str
    kind: str
    name: str
    symbol_path: str
    locator: dict[str, Any]
    report_path: str


@dataclass
class QueryResult:
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    used_claim_ids: list[str] = field(default_factory=list)
    used_verbatim_spans: list[VerbatimSpan] = field(default_factory=list)
    used_structure_items: list[StructureItemHit] = field(default_factory=list)
    # Slice 077: additive ``ids_only`` flag. When True, ``answer``
    # is empty and ``used_verbatim_spans`` / ``used_structure_items``
    # stay empty; ``used_claim_ids`` carries the filtered + ranked
    # claim ids the CLI emits one per line. ``citations`` is still
    # populated so callers that ask for ids-only can still inspect
    # the structured citation metadata if they want it.
    ids_only: bool = False


@dataclass
class MergeProposalSummary:
    proposal_id: str
    source_entity_id: str
    target_entity_id: str
    status: str
    proposed_alias_text: str


@dataclass
class StatusResult:
    """Compact workspace health summary.

    The ``lock_*`` fields beyond ``lock_held`` / ``lock_owner`` are
    additive (Slice 069). They surface enough metadata for an agent
    to choose between waiting on a live operation, running
    ``llloom unlock --clear-stale --reason "..."``, running
    ``llloom reconcile``, or escalating an unrecoverable stale lock
    for manual investigation.
    """

    source_count: int
    claim_count: int
    rendered_page_count: int
    pending_review_count: int
    stale_count: int
    retracted_count: int
    lock_held: bool
    lock_owner: str | None
    last_operation_id: str | None
    last_operation_status: str | None
    lock_op_id: str | None = None
    lock_acquired_at: str | None = None
    lock_heartbeat_at: str | None = None
    lock_timeout_seconds: int | None = None
    lock_is_timed_out: bool = False
    lock_recoverable: bool = False
    lock_recoverability_reason: str | None = None
    recommended_lock_action: str | None = None
    # Slice 085: additive lock-owner-process diagnostics. ``None`` when
    # no lock is held or when the lock has no owner metadata; otherwise
    # mirror the optional ``Lock.owner_*`` fields. ``lock_owner_pid_state``
    # is the result of :func:`llloom.state.lock.local_owner_pid_state`
    # — one of ``"alive"`` / ``"dead"`` / ``"unknown"`` when a lock
    # exists, ``None`` when no lock is held. The recoverability /
    # recommended-action fields above continue to be governed by the
    # frozen timeout + journal predicate, never by PID state.
    lock_owner_pid: int | None = None
    lock_owner_hostname: str | None = None
    lock_owner_cwd: str | None = None
    lock_owner_command: str | None = None
    lock_owner_pid_state: str | None = None


@dataclass(frozen=True)
class RenderTargetSummary:
    """One ``(page_id, block_id)`` render target attached to a claim.

    Slice 077 added this dataclass for the read-only inspection
    surface (``llloom list-claims``, ``llloom claim-card``). The
    frozen pair mirrors the existing ``Assertion.render_targets``
    list element shape so a serialized card or claim summary
    surfaces the same identity an agent would write back.
    """

    page_id: str
    block_id: str


@dataclass(frozen=True)
class EvidenceSummary:
    """Bounded evidence summary for a single claim's source span.

    Slice 077 added this dataclass for the read-only claim card.
    ``locator`` is the locator mapping (same shape as
    ``Locator.to_mapping()``). ``excerpt_hash`` is the stored
    ``sha256:`` hash; ``excerpt_preview`` is a bounded preview
    (240 chars, same bound as the verifier mismatch preview)
    of the stored ``excerpt`` when present — never a raw source
    body.
    """

    source_id: str
    locator: dict[str, Any]
    excerpt_hash: str
    excerpt_preview: str | None = None


@dataclass(frozen=True)
class ClaimSummary:
    """One-line claim summary for ``list-claims`` output.

    Slice 077 added this dataclass for the read-only listing
    surface. ``qualified_target`` is ``claim:<entity_id>:<claim_id>``
    so the same identity can be passed straight to
    ``claim-card`` or ``promote``.
    ``claim_text_preview`` is bounded to 240 characters so listing
    a large workspace never emits unbounded source-grounded text.
    """

    qualified_target: str
    claim_id: str
    entity_id: str
    entity_display_name: str
    entity_type: str
    claim_kind: str
    status: str
    verification_status: str
    supersedes: str | None
    source_ids: list[str] = field(default_factory=list)
    render_targets: list[RenderTargetSummary] = field(default_factory=list)
    claim_text_preview: str = ""


@dataclass(frozen=True)
class ClaimCard:
    """Detailed read-only inspection record for one claim.

    Slice 077 added this dataclass for ``llloom claim-card``. The
    card includes everything an agent needs to understand a claim
    without opening YAML by hand. Evidence summaries carry bounded
    excerpt previews; the card itself is purely read-only and is
    never persisted.
    """

    qualified_target: str
    claim_id: str
    entity_id: str
    entity_display_name: str
    entity_type: str
    claim_kind: str
    claim_text: str
    status: str
    verification_status: str
    supersedes: str | None
    evidence: list[EvidenceSummary] = field(default_factory=list)
    render_targets: list[RenderTargetSummary] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class SourceSummary:
    """Read-only source registry record summary.

    Slice 077 added this dataclass for ``llloom list-sources``.
    Reports registry metadata only — never raw source bodies.
    ``content_hash`` and ``byte_size`` mirror the registry record;
    ``retracted_at`` / ``retraction_reason`` are populated for
    retracted sources.
    """

    source_id: str
    source_class: str
    raw_path: str
    content_hash: str
    byte_size: int
    status: str
    registered_at: str = ""
    last_seen_at: str = ""
    retracted_at: str | None = None
    retraction_reason: str | None = None


@dataclass(frozen=True)
class PageSummary:
    """Read-only page summary for ``list-pages``.

    Slice 077 added this dataclass. ``page_id`` is the value
    declared in the page's YAML frontmatter (falling back to the
    file stem when the frontmatter omits it). ``page_path`` is the
    workspace-relative POSIX path. ``page_class``, ``status``, and
    ``write_policy`` are frontmatter mirrors — left empty when the
    frontmatter omits a value or the page parses cleanly but lacks
    that field.
    """

    page_id: str
    page_path: str
    page_class: str = ""
    status: str = ""
    write_policy: str = ""


@dataclass(frozen=True)
class RenderTargetListEntry:
    """One render-target row for ``list-render-targets``.

    Slice 077 added this dataclass. Reuses the Slice 073
    read-only render-plan discovery without acquiring any lock,
    opening any journal entry, or writing any page / fingerprint /
    transaction directory. ``contributing_claim_ids`` is the
    sorted list of claim ids that would render into the block;
    ``marker_health`` mirrors :class:`RenderPlanEntry.marker_health`
    (``ok`` / ``missing_page`` / ``parse_error``).
    """

    page_id: str
    block_id: str | None
    page_path: str
    marker_health: str
    marker_message: str | None
    contributing_claim_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DoctorWarning:
    """One read-only health-diagnostic warning from ``doctor(...)``.

    Slice 079 added this dataclass. ``warning_id`` is a stable
    deterministic key (e.g. ``"render:fingerprint-drift:concept/foo"``)
    so accepted-warning allowlists can match exactly; ``severity`` is
    ``"info"``, ``"warning"``, or ``"error"``; ``category`` groups
    related warnings (`lock`, `render`, `sidecar`, `structure-report`,
    `transaction`, `lifecycle`, `source`, `page`,
    `accepted-warnings`); ``message`` is operator-facing prose;
    ``recommended_command`` (when present) names the canonical
    single-command repair (or ``None`` when no one-shot repair
    exists — the operator-facing message must say so honestly);
    ``evidence`` is a list of bounded references (workspace-relative
    paths, op ids, claim ids) — never raw source bodies.
    """

    warning_id: str
    severity: str
    category: str
    message: str
    recommended_command: str | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AcceptedDoctorWarning:
    """One warning matched against the accepted-warning allowlist.

    Slice 079 added this dataclass. ``warning`` carries the full
    :class:`DoctorWarning` payload so reviewers see the same
    structured signal an unaccepted warning would surface;
    ``accepted_reason`` is the non-empty justification from the
    allowlist entry; ``accepted_by`` / ``accepted_at`` are optional
    audit fields (``None`` when the allowlist omits them);
    ``evidence_links`` is the per-entry evidence path list (at
    least one entry; missing evidence refuses the acceptance and
    keeps the warning on the unaccepted list with a separate
    ``accepted-warnings:malformed-entry`` warning).
    """

    warning: DoctorWarning
    accepted_reason: str
    accepted_by: str | None = None
    accepted_at: str | None = None
    evidence_links: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UpdateReviewBundle:
    """Memory-update review bundle for one operation id.

    Slice 079 added this dataclass. Produced by
    ``doctor(workspace, op_id=...)`` / ``doctor(workspace,
    last_op=True)``. The bundle reads the
    ``state/journals/<op_id>.yaml`` entry, optional
    ``state/reports/updates/<op_id>.yaml`` seed-apply report,
    and the current lint / verify / status summaries. Text
    fields are bounded; raw source bodies, full page text, and
    full claim text are never embedded.
    """

    op_id: str
    op_kind: str
    journal_status: str
    started_at: str
    completed_at: str | None = None
    touched_files: list[str] = field(default_factory=list)
    seed_update_report_path: str | None = None
    source_changes: list[str] = field(default_factory=list)
    claim_changes: list[str] = field(default_factory=list)
    rendered_pages: list[str] = field(default_factory=list)
    lint_summary: dict[str, int] = field(default_factory=dict)
    verify_summary: dict[str, int] = field(default_factory=dict)
    status_summary: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[DoctorWarning] = field(default_factory=list)
    accepted_warnings: list[AcceptedDoctorWarning] = field(default_factory=list)


@dataclass
class DoctorResult:
    """Result of ``doctor(workspace, ...)`` and ``llloom doctor``.

    Slice 079 introduced this dataclass. ``warnings`` lists every
    current diagnostic the doctor surfaced that did NOT match an
    accepted-warning allowlist entry; ``accepted_warnings`` lists
    warnings the allowlist explicitly accepted (with evidence
    links); ``stale_acceptances`` lists allowlist entries whose
    ``warning_id`` did not match any current diagnostic — the
    listing is a "you can remove this entry now" hint.
    ``recommended_next_commands`` aggregates the
    :attr:`DoctorWarning.recommended_command` strings from the
    unaccepted ``warnings`` in stable order (no duplicates). The
    ``update_review`` field is populated only when the caller
    asked for an operation-scoped bundle.
    """

    target: str = "doctor"
    warning_count: int = 0
    accepted_warning_count: int = 0
    warnings: list[DoctorWarning] = field(default_factory=list)
    accepted_warnings: list[AcceptedDoctorWarning] = field(default_factory=list)
    stale_acceptances: list[str] = field(default_factory=list)
    recommended_next_commands: list[str] = field(default_factory=list)
    update_review: UpdateReviewBundle | None = None


@dataclass
class HealthReport:
    """Drift report from ``rebuild(..., target="health_report")``.

    Detection-only and derived; never repairs. Every field is recomputed
    on each rebuild from existing workspace state and persists to
    ``state/reports/health/health_report.yaml``. The report itself is
    derived: deletable without data loss, regenerated by re-running the
    rebuild.

    Field semantics:

    - ``target`` is always the literal ``"health_report"``.
    - ``search_sidecar`` and ``graph_sidecar`` are presence-only strings,
      one of ``"present"`` or ``"missing"``. Stale-row validation belongs
      to the search/graph subsystems themselves at query time, not here.
    - ``interrupted_journals`` lists ``op_id`` values whose journal entry
      is still ``in_progress`` but no currently-held lock claims that
      ``op_id``. These are candidates for ``reconcile``; the report
      never clears them.
    - ``stale_lock_unrecoverable`` is True iff a lock is currently held,
      the lock is timed out, AND
      :meth:`llloom.state.lock.WorkspaceLock.is_stale_recoverable`
      returns ``(False, ...)``. This is the explicit "manual attention
      needed" signal — ``reconcile`` will refuse to clear such a lock.
    - ``missing_structure_reports`` lists ``source_id`` values whose
      schema policy resolves to ``structure_extract`` but whose
      ``state/structure/<source_id>.yaml`` file is absent. Distinct
      from ``structure_report_drift`` (file present but stale).
    - ``structure_report_drift`` lists ``source_id`` values whose report
      exists but is unusable as a current derived report (mismatched
      ``content_hash``, mismatched ``source_class``, wrong ``version``,
      malformed YAML, or wrong top-level shape).
    - ``render_fingerprint_drift`` lists ``page_id`` values whose stored
      fingerprint disagrees with the fingerprint recomputed from current
      authoritative claims (missing-stored, missing-recomputed, and
      differing-values all count). Sorted deterministically.
    """

    target: str
    entity_count: int
    claim_count: int
    pending_proposals: int
    stale_claims: int
    retracted_claims: int
    lock_held: bool
    lock_owner: str | None
    interrupted_journals: list[str] = field(default_factory=list)
    stale_lock_unrecoverable: bool = False
    search_sidecar: str = "missing"
    graph_sidecar: str = "missing"
    structure_report_drift: list[str] = field(default_factory=list)
    missing_structure_reports: list[str] = field(default_factory=list)
    render_fingerprint_drift: list[str] = field(default_factory=list)
