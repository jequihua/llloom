# Public API Contract

## Library API

```python
from llloom import Workspace
from llloom.ops import (
    ingest,
    verify,
    render,
    query,
    lint,
    reconcile,
    unlock,
    promote,
    retract,
    rebuild,
    list_merge_proposals,
    review_alias,
    merge_alias,
    reject_alias,
    status,
    prepare_pdf,
)
```

The optional `prepare_pdf` operation lives in
`llloom.ops.prepare_pdf`. The Docling adapter
(`llloom.pdf_prep.convert_with_docling`) lazy-imports `docling`
inside the function; importing `llloom`, `llloom.cli`, or
`llloom.pdf_prep` does not require the `llloom[docling]` extra.
Callers can inject a custom `adapter` callable to bypass Docling
entirely (used by tests). Result type: `PdfPrepResult`.

Every operation returns a dataclass from `llloom.ops.results`:

- `IngestResult` -- carries `render_skipped: bool` (default False),
  set true when ``no_render=True`` was passed to a
  `claim_extract_and_view_render` ingest so the render step was
  deliberately suppressed. Verified claims still persist.
  `extraction_errors: list[str]` records per-candidate parse or
  verifier failures on a refused (batch-atomic) extraction; for
  ``excerpt_hash`` mismatches the entry includes the structured
  diagnostic fields (source id, locator type, both hashes, bounded
  preview). Slice 070 changed `claims_created: list[str]` to
  `claims_created: list[CreatedClaim]` — every persisted claim
  surfaces as a `CreatedClaim(claim_id, entity_id, status,
  verification_status)` record so the create result shows the
  lifecycle authority at the moment of creation. The CLI
  serializes the list as JSON objects through the existing
  dataclass recursion.
- `CreatedClaim` (Slice 070) — frozen dataclass returned inside
  `IngestResult.claims_created`. Fields: `claim_id: str`,
  `entity_id: str`, `status: str` (the persisted
  `Assertion.status` — `"draft"` by default for deterministic
  seeds, the model parser's status for model-backed candidates,
  or whatever the per-candidate `SeedClaim.status` override
  named), `verification_status: str` (always `"verified"` on a
  persisted record; any verifier failure refuses the whole
  batch).
- `VerifyResult` -- carries `mismatches: list[VerifierMismatch]`
  alongside the existing `notes: list[str]`. Each `VerifierMismatch`
  holds `claim_id`, `source_id`, `locator_type`, `stored_hash`,
  `computed_hash`, `current_preview`, and optional `stored_preview`
  (from the persisted evidence excerpt). Previews are bounded to a
  fixed maximum (currently 120 characters, whitespace-collapsed,
  truncated with `...`) and are intended for human or agent review;
  they are not a back-channel for raw source bodies.
- `RenderResult` (aliased as `OpsRenderResult` where it would collide
  with the internal `pages.render.RenderResult`). `render(workspace,
  target=...)` validates `target` in a lockless preflight before
  acquiring the workspace lock or starting a render journal entry:
  accepted forms are `None` (render every claim-referenced page) and
  `"page:<page_id>"`; any other form raises `ValueError` cheaply
  with a hint pointing at the accepted syntax. The CLI converts a
  preflight `ValueError` into a stderr message and exit code 1.
  Slice 071 moved render planning to page/block-centric: every page
  is rendered at most once per operation and the rendered claim
  block is the union of every entity's render-visible assertions
  targeting `(page_id, block_id)`, ordered by
  `(entity_id, claim_id)`. `rendered_pages` and `unchanged_pages`
  contain at most one entry per page id; the stored render
  fingerprint is the union fingerprint. New public helpers in
  `llloom.pages.render` —
  `render_claim_block_from_contributors(...)`,
  `compute_render_fingerprint_from_contributors(...)`,
  `render_page_file_from_contributors(...)`, and
  `compute_page_render_fingerprints(...)` — back the union path.
  Slice 073 added additive `RenderResult` fields `plan:
  list[RenderPlanEntry]`, `dry_run: bool = False`, and
  `list_targets: bool = False`, plus the public
  `RenderPlanContributor(entity_id, display_name, claim_ids)` and
  `RenderPlanEntry(target, page_id, page_path, block_id,
  contributors, contributing_claim_ids, marker_health,
  marker_message, content_would_change, fingerprint_would_change,
  planned_fingerprint, stored_fingerprint)` dataclasses.
  `render(workspace, *, target=None, dry_run=False,
  list_targets=False)` runs read-only (no
  `WorkspaceLock.acquire`, no journal entry, no page write, no
  fingerprint write) when either flag is True and populates
  `result.plan`; defaults preserve byte-identical mutating
  behavior. The CLI exposes the flags as `llloom render
  [--dry-run] [--list-targets] [page:<page_id>]`. Marker health is
  one of `ok` / `missing_page` / `parse_error`; the read-only
  paths surface parse errors via `marker_health` while the
  mutating path still raises `RenderError`. Slice 074 added the
  stage-then-commit primitive
  `llloom.state.render_transactions.RenderTransaction` and the
  new `Workspace.state_transactions` property; mutating `render`
  stages every page and the full
  `state/render_fingerprints.yaml` snapshot under
  `state/transactions/<op_id>/` before committing them together,
  and removes the transaction directory on success. Pre-commit
  failure leaves final pages + fingerprints byte-identical;
  failure during commit leaves a diagnosable transaction
  manifest on disk while the operation's journal stays
  `in_progress` and its lock stays held for `reconcile`. Dry-run
  and list-targets paths create no transaction directory.
  `reconcile` cleans an abandoned render transaction directory
  only when the same call is clearing the matching stale lock via
  the Slice 069 journal-backed recoverability predicate.
  Legacy single-entity helpers (`render_claim_block(entity, ...)`,
  `compute_render_fingerprint(entity, ...)`,
  `render_page_file(..., entity, ...)`) keep working; the
  non-empty single-entity output is byte-identical to the
  pre-Slice-071 shape. The empty single-entity case (every
  block-targeting assertion filtered out by the render-visible
  rule) is restored by Slice 071a to the pre-Slice-071 sentinel
  shape `## <display_name>\n\n_No rendered assertions._\n` —
  `render_claim_block(entity, ...)` post-processes the union
  helper's bare-sentinel output and `render_page_file(..., entity,
  ...)` routes its claim-block render through the wrapper so the
  on-disk page preserves the legacy empty shape. The union helpers
  themselves keep returning the bare `_No rendered assertions._\n`
  sentinel for the multi-entity empty case (no contributor's
  display name would be representative); Slice 071a's contract
  tests pin this divergence.
- `QueryResult` — carries `used_verbatim_spans: list[VerbatimSpan]`
  (deterministic exact retrieval from registered `index_only`
  sources; empty when no source matched) and
  `used_structure_items: list[StructureItemHit]` (structure-report
  items surfaced through the search sidecar and rehydrated from
  the on-disk report under `state/structure/<source_id>.yaml`;
  empty when the sidecar is absent or no structure item matched).
  Every existing field (`question`, `answer`, `citations`,
  `used_claim_ids`) is preserved. Slice 077 added the additive
  `ids_only: bool` flag (default `False`); when `True` the
  ``answer`` is empty and ``used_verbatim_spans`` /
  ``used_structure_items`` stay empty. Slice 077a pinned the
  empty-question contract: ``query(ws, question="")`` with all
  knobs at defaults returns no citations and no used ids (matches
  the pre-Slice-077 behavior); empty-token admission requires
  explicit inspection (any Slice 077 filter knob or
  ``ids_only=True``). Each citation dict now
  carries `entity_id`, `claim_id`, `claim_text` (bounded
  preview), `claim_kind`, `status`, `verification_status`,
  `supersedes` (when present), `source_ids`, and
  `render_targets` — never raw source bodies.
- `ClaimSummary` / `ClaimCard` / `EvidenceSummary` /
  `RenderTargetSummary` / `SourceSummary` / `PageSummary` /
  `RenderTargetListEntry` (Slice 077, from `llloom.ops`) — the
  read-only inspection dataclasses returned by `list_claims`,
  `claim_card`, `list_sources`, `list_pages`, and
  `list_render_targets`. Every record carries identity (qualified
  claim target / source_id / page_id) plus the metadata an agent
  needs to inspect authority without opening YAML by hand.
  Text fields are bounded at 240 characters; no raw source body
  is emitted. `claim_card` accepts either the qualified form
  `claim:<entity_id>:<claim_id>` or a bare `claim_id` (only when
  unique); missing / ambiguous bare ids raise
  `ClaimCardError` (CLI exits 1 with a stderr diagnostic that
  lists every candidate qualified target). Filters live on
  `llloom.ops.query.query(...)` and on the list operations;
  unknown filter values raise
  `llloom.ops.query.QueryFilterError` /
  `llloom.ops.inspect.InspectFilterError`. The CLI exposes the
  surface as `llloom query ... [--status STATE]... [--ids-only]`,
  `llloom list-claims ...`, `llloom claim-card TARGET`,
  `llloom list-sources ...`, `llloom list-pages [--ids-only]`,
  and `llloom list-render-targets [--page PAGE_ID] [--ids-only]`
  (CLI verb count 18 → **23**).
- `VerbatimSpan` — `source_id`, `excerpt`, `excerpt_hash` (SHA-256 of
  `excerpt`), `char_start`, `char_end`. `excerpt` is contiguous raw
  source text; `excerpt == raw_source[char_start:char_end]` always
  holds. The hash matches the convention used by the harness's
  `SourceSpan` typed input class so callers can correlate without
  re-hashing.
- `StructureItemHit` — `source_id`, `source_class`, `language`,
  `kind`, `name`, `symbol_path`, `locator: dict`, `report_path`.
  Produced by `query(...)` only when the search sidecar is
  present and a structure-report item matches. Every field is
  rehydrated from the on-disk report under
  `state/structure/<source_id>.yaml` before emission; the
  SQLite row is used only as a candidate filter. The `locator`
  is a `code_v1` mapping; scalar values, comments, docstrings,
  and code bodies are never carried.
- `LintResult` -- `canary_hits: list[str]` is populated whenever the
  canary scan finds the fixed fixture token, any caller-supplied
  extra token, or (when `lint(..., generated_canary=True)` is used)
  a freshly generated per-run token inside a forbidden observation
  point. `generate_canary_token()` in `llloom.ops.lint` returns
  stdlib-only per-run tokens prefixed `LLLOOM_CANARY_RUN_`.
- `ReconcileResult`
- `UnlockRecord` — Slice 069 made `unlock` truthful. The dataclass
  carries `mode: str` (`"unlock_window"` for the legacy
  journal-only window; `"clear_stale_lock"` for the guarded
  stale-lock clear; Slice 086 adds `"clear_dead_owner_lock"` for
  the guarded local same-host dead-owner clear),
  `lock_cleared: bool` (always `False` for `unlock_window`;
  `True` only when the corresponding predicate held: for
  `clear_stale_lock` that is `is_stale_recoverable(lock,
  journal=journal)`, for `clear_dead_owner_lock` that is the
  Slice 086 nine-step local predicate — same host + `owner_pid > 0`
  + matching `socket.gethostname()` + `local_owner_pid_state(lock)
  == "dead"` + lock not yet timed out + matching journal entry
  exists + entry is `in_progress` + entry has no `completed_at` +
  identical pre-clear re-read), `refused: bool` +
  `refusal_reason: str | None`, the `prior_op_id` /
  `prior_owner_id` / `prior_acquired_at` / `prior_heartbeat_at`
  quartet that captures the cleared lock's identity on the
  success path, and the Slice 086 additive fields
  `prior_owner_pid: int | None` / `prior_owner_hostname: str | None`
  / `prior_owner_pid_state: str | None` (populated on the
  `clear_dead_owner_lock` success path with the dead PID,
  the matching local hostname, and the constant `"dead"`).
  `--dead-owner` is mutually exclusive with `--clear-stale`,
  workspace-only (does not accept a positional target — passing
  one refuses cleanly at both the CLI and library layers
  before any lock or journal is touched, per Slice 086a), and
  the operator escape hatch is local same-host only and never
  bypasses the frozen stale-recovery rule. The CLI exits `1`
  on refusal (clean stderr) and `0` on success.
- `PromoteResult` — Slice 078 wired the result's `reason` field
  to `llloom.claims.lifecycle.explain_transition_refusal(...)` so
  illegal transitions surface a concrete promotion-path
  suggestion (e.g. `"promote through 'reviewed' first"`). The
  legal transition graph lives in
  `LEGAL_LIFECYCLE_TRANSITIONS` and is shared with
  `supersede(...)`; `ops.promote.ALLOWED_TRANSITIONS` is now a
  back-compat alias for the canonical frozenset.
- `SupersedeResult` (Slice 078) — result of
  `supersede(workspace, *, old, by)` and `llloom supersede
  OLD --by NEW`. Carries `old_target`, `new_target` (both
  fully-qualified `claim:<entity_id>:<claim_id>` strings),
  `old_from_status`, `old_to_status` (constant `"superseded"`
  on success), `new_status` (always `"validated"` on success
  because the slice requires the replacement to be already
  validated), `supersedes` (the qualified OLD target written
  to `Assertion.supersedes`), `op_id`, `refused`, `reason`.
  Target resolution accepts either `claim:<entity>:<claim>`
  or a bare `claim_id` (only when unique across the
  workspace); missing / ambiguous bare ids raise
  `SupersedeError` (CLI exits 1 with a stderr diagnostic that
  lists every candidate qualified target). The operation is
  atomic under the existing `operation(...)` lock / journal
  contract — pre-mutation refusals leave every entity YAML
  byte-identical; success saves every touched entity exactly
  once before the context closes. The CLI exposes the
  operation as `llloom supersede <old-claim-target-or-id>
  --by <new-claim-target-or-id>` (CLI verb count 23 → **24**).
- `DoctorResult` / `DoctorWarning` / `AcceptedDoctorWarning` /
  `UpdateReviewBundle` (Slice 079) — result types for
  `doctor(workspace, *, op_id=None, last_op=False,
  accepted_warnings=None)` and `llloom doctor [--op-id OP_ID
  | --last-op] [--accepted-warnings PATH]`. `DoctorResult`
  carries `warning_count`, `accepted_warning_count`,
  `warnings: list[DoctorWarning]`, `accepted_warnings:
  list[AcceptedDoctorWarning]`, `stale_acceptances`,
  `recommended_next_commands`, and (optional)
  `update_review: UpdateReviewBundle`. Each `DoctorWarning`
  has a stable deterministic `warning_id` (e.g.
  `"render:fingerprint-drift:concept/foo"`), a `severity`
  (`info` / `warning` / `error`), a `category`, a `message`,
  an optional `recommended_command`, and bounded `evidence`
  references (workspace paths / op ids / claim ids — never
  raw source bodies). The accepted-warning matching is
  exact on `warning_id`; entries with missing `reason` or
  empty `evidence` are not accepted (a separate
  `accepted-warnings:malformed-entry:<path>` diagnostic
  surfaces). `UpdateReviewBundle` is produced only when the
  caller asks for an operation-scoped bundle; it carries
  the journal entry's identity, the optional Slice 076
  seed update report's `source_changes` / `claim_changes`
  / `rendered_pages` / `provenance` extracts, and current
  `lint` / `verify` / `status` count-only digests. The
  operation is **strictly read-only**: no `operation(...)`
  context, no workspace lock acquisition, no journal
  entry, no sidecar / page / fingerprint / claim / source
  / report / transaction write, no model / provider call.
  Documented exit-code contract on the CLI: `0` when no
  unaccepted `error`-severity warnings AND no
  `review-bundle:*` warnings; `1` otherwise (so
  `llloom doctor` is a usable shell precondition gate for
  `seed apply` / `ingest` / `render`). See
  `04_specification/operations_and_cli.md` `## doctor` for
  the warning-category list, accepted-warning grammar, and
  bundle shape. CLI verb count 24 → **25**.
- `PageCreateResult` (Slice 084) — result of `create_page(
  workspace, *, page_id, page_class=None, title=None)` /
  `llloom page create <page_id> [--page-class CLASS]
  [--title TITLE]`. Fields: `page_id` (the normalized page
  id as written; reflected in the YAML frontmatter),
  `page_class` (one of `entity` / `concept` / `synthesis` /
  `navigation`), `page_path` (workspace-relative POSIX of
  the created stub, e.g. `pages/concepts/foo.md`),
  `claim_block_id`, `commentary_id` (deterministic marker
  ids: slashes replaced with dots, prefixed with
  `claim_block.` / `commentary.`), `status` (constant
  `"draft"` on the success path), `op_id` (the
  `op_kind="page_create"` journal entry id when the
  operation context opened; empty for pre-operation
  refusals), and `refusal_reason` (set when a refusal is
  detected inside the operation context). Pre-operation
  refusals (malformed `page_id`, unknown `--page-class`,
  conflicting inferred / explicit class, existing target
  file, path-traversal / class-directory escape) raise
  `PageCreateError` and never open a journal entry. The
  operation is mutating but never creates claims, never
  invokes a model, and never updates render fingerprints —
  it writes exactly one Markdown file under
  `pages/<class_dir>/<tail>.md`. No overwrite flag in this
  slice: existing pages refuse with exit code 1. Path
  resolution strips a recognized class prefix from the
  `page_id` when present (`concept/foo` →
  `pages/concepts/foo.md` with `page_id: concept/foo` in
  frontmatter); inferring `--page-class` from the prefix is
  the smaller-friction path, otherwise pass it explicitly.
  CLI verb count 25 → **26**.
- `RetractResult` â€” `affected_pages` is a list of **workspace-relative
  POSIX paths** (e.g. `pages/concepts/foo.md`), not page IDs. Same
  shape as `rerendered_pages`. See
  `04_specification/operations_and_cli.md` Â§retract.
- `MergeProposalSummary`
- `SeedManifestResult` / `PlannedSeedClaim` (Slice 075) — the
  result and planned-claim records for `apply_seed_manifest(...)`
  and `llloom seed apply`. `SeedManifestResult` carries
  `manifest_path`, `dry_run`, `no_render`, `sources_planned`,
  `claims_planned: list[PlannedSeedClaim]` (effective merged
  status per claim, populated on dry-run + real apply),
  `claims_created: list[CreatedClaim]` (real-apply success
  only), `entities_touched`, `pages_rendered`,
  `render_skipped`, `refusal_reason`, and `op_ids` (one
  `op_kind="seed_apply"` journal op id per persisted source).
  The CLI exposes the operation as `llloom seed apply
  <manifest.yaml> [--dry-run] [--no-render] [--status
  <status>]`. The apply path is deterministic and model-free
  by construction: it routes through the existing
  `_apply_candidates(...)` verifier + atomic-persistence
  primitive (with the new `dry_run=False` / `dry_run=True`
  knob) and never invokes `LLMInvoke` / `NullModel` / any
  model provider. The journal entry carries `invocation_logs:
  []` and notes naming the deterministic-no-provider mode.
  See `04_specification/seed_manifest_v1.md` for the manifest
  schema, merge order, refusal rules, and dry-run /
  no-render guarantees.
  Slice 076 added additive fields: `report_path: str | None`
  (workspace-relative POSIX path of the durable update report
  at `state/reports/updates/<op_id>.yaml`; `None` on dry-run /
  refusal), `excerpt_checks: list[SeedExcerptCheck]` (one
  entry per claim recording the `excerpt_equality` decision,
  bounded preview, and hash), and `counts_before` /
  `counts_after: dict[str, int]` (workspace-wide source /
  entity / claim / page totals captured around the apply).
- `SeedExcerptCheck` (Slice 076, from `llloom.state`) — frozen
  dataclass carrying `claim_id`, `mode` (`none` or
  `exact_one_sentence`), `matched`, `excerpt_hash`,
  `excerpt_preview` (bounded at 240 characters), and optional
  `message` (set only on a mismatch). Produced by the pure
  helper `check_seed_excerpt_equality(...)` which the seed
  apply path runs against the locator-resolved excerpt for
  every claim. Re-exported from `llloom.state.__init__`
  alongside `write_seed_update_report(...)` and
  `SEED_UPDATE_REPORT_VERSION = "seed_update_report_v1"`. The
  helper is I/O-free and never invokes a model.
- `StatusResult` — Slice 069 added an additive lock-recoverability
  block: `lock_op_id`, `lock_acquired_at`, `lock_heartbeat_at`,
  `lock_timeout_seconds`, `lock_is_timed_out: bool`,
  `lock_recoverable: bool`, `lock_recoverability_reason: str | None`,
  and `recommended_lock_action: str | None`. Existing fields
  (`lock_held`, `lock_owner`, `last_operation_id`,
  `last_operation_status`, and the six count fields) keep their
  shape; every new field has a default so older constructors keep
  working. `recommended_lock_action` is a human-readable hint
  (`"wait for op_id=... or contact the lock owner"`,
  `'llloom unlock --clear-stale --reason "..."  (or: llloom
  reconcile)'`, `"manual investigation required: ..."`) and is
  `None` when no lock is held.
  Slice 085 added five additive lock-owner-process
  diagnostic fields: `lock_owner_pid: int | None`,
  `lock_owner_hostname: str | None`,
  `lock_owner_cwd: str | None`,
  `lock_owner_command: str | None`, and
  `lock_owner_pid_state: str | None` (one of `"alive"` /
  `"dead"` / `"unknown"` when a lock is held; `None`
  otherwise). The PID-state classification routes through
  the new `llloom.state.lock.local_owner_pid_state(...)`
  helper which is conservative by construction: only
  `"alive"` / `"dead"` on confident same-host
  `os.kill(pid, 0)` evidence. `PermissionError` and
  `OSError` with `errno.EPERM` are classified `"alive"`
  because the OS is confirming the process exists but the
  current user cannot signal it (a missing process raises
  `ProcessLookupError` / ESRCH and routes to `"dead"`);
  `"unknown"` is reserved for missing metadata, hostname
  mismatch, `owner_pid <= 0`, non-EPERM OSErrors, platform
  uncertainty, and unexpected exceptions. The PID-state
  field is **diagnostic only** — it never widens
  `lock_recoverable` or `recommended_lock_action`. The frozen stale-recovery rule
  (`timeout elapsed + matching in-progress journal
  evidence`) is unchanged. The lock file shape (Lock YAML
  under `state/locks/workspace.yaml`) gains four optional
  fields (`owner_pid`, `owner_hostname`, `owner_cwd`,
  `owner_command`); old lock YAML omitting these is parsed
  cleanly with the fields set to `None` and is NOT reported
  as `lock:malformed` by `doctor`. A doctor warning
  category entry `lock:owner-process-dead:<op_id>`
  (severity `warning`, category `lock`) is emitted only
  for a same-host confidently-dead-PID lock that is NOT
  yet timed out. Slice 086 made the recommended command
  conditional: when the matching journal entry exists and
  is `in_progress` with `completed_at is None`, the
  recommendation reads
  `llloom unlock --dead-owner --reason "..."` (the
  guarded local operator escape hatch is safe to take);
  otherwise the recommendation falls back to wait /
  `llloom reconcile` / `llloom unlock --clear-stale
  --reason "..."` and the literal token `--dead-owner` is
  deliberately omitted from the fallback so the operator
  cannot misread it as a suggestion. The warning never
  tells the operator to force-clear the lock — `doctor`
  remains strictly read-only and never performs any file
  write.
- `HealthReport` — deterministic read-only drift report returned by
  `rebuild(workspace, target="health_report")`. Detection only; no
  remediation, no auto-rebuild, no auto-unlock. Fields: `target`
  (always `"health_report"`), `entity_count`, `claim_count`,
  `pending_proposals`, `stale_claims`, `retracted_claims`,
  `lock_held`, `lock_owner`, `interrupted_journals: list[str]`
  (in-progress journal `op_id`s whose lock is not currently held —
  candidates for `reconcile`), `stale_lock_unrecoverable: bool`
  (lock is currently held, timed out, and `is_stale_recoverable`
  returns False — the explicit "manual `unlock`/attention needed"
  signal), `search_sidecar` and `graph_sidecar` ∈ {`"present"`,
  `"missing"`} (presence-only; rebuild via `rebuild search` /
  `rebuild graph`), `missing_structure_reports: list[str]` (source
  ids whose schema policy resolves to `structure_extract` but whose
  `state/structure/<source_id>.yaml` report is absent — re-ingest
  to produce it), `structure_report_drift: list[str]` (source ids
  whose report exists but no longer matches the registry's
  `content_hash`, `source_class`, version, or top-level shape —
  re-ingest), `render_fingerprint_drift: list[str]` (page ids whose
  stored fingerprint disagrees with the fingerprint recomputed from
  current authoritative claims — re-run `render` or
  `rebuild render_fingerprints`). The deepened report writes
  deterministically to
  `state/reports/health/health_report.yaml` via
  temp-file-and-rename. It does **not** acquire the workspace lock
  (so it remains callable when the workspace is locked, including
  the stale-unrecoverable case it must report on); it writes its
  own completed journal entry directly for audit parity.
  Malformed lock files on this path (YAML-parse-corrupt content or
  YAML-valid mappings missing required `Lock` keys) surface as
  **held-with-no-owner**: `lock_held=True`, `lock_owner=None`. The
  report never silently downgrades a corrupt lock to "no lock at
  all". This is enforced symmetrically inside
  `WorkspaceLock.read()`, which now wraps both YAML-parse failures
  and shape-invalid mappings in `LockError` so every caller
  (including `reconcile`) treats malformed locks consistently.
- `PdfPrepResult` — produced by the optional `prepare_pdf`
  operation. Fields: `prep_id`, `status` (`succeeded` | `failed` |
  `refused`), `bundle_dir` (workspace-relative POSIX),
  `manifest_path`, `source_pdf`, `source_pdf_sha256`, `provider`
  (`docling_default`), `artifacts: list[PdfPrepArtifact]`,
  `components: dict[str, str]`, `selected_artifact: PdfPrepArtifact
  | None`, `refusal_reason: str | None`, `op_id`. The `succeeded`
  property is true iff `status == "succeeded"`. On refusal /
  failure, `selected_artifact` is `None` and `refusal_reason`
  carries the explanation surfaced to the CLI.
- `PdfPrepArtifact` — one artifact entry in a prep bundle.
  Fields: `path` (workspace-relative POSIX), `kind` (e.g.
  `docling_markdown`, `docling_json`), `sha256` (`sha256:<hex>`).

## Seed-claim entry point

`ingest` accepts an optional `seed_claims: list[SeedClaim]` parameter.
Seed claims are the deterministic path used by tests and by callers who
already have structured claim candidates. Seed claims are still routed
through `LLMInvoke` for audit parity with model-extracted claims, and
each seed claim passes the span verifier before persistence.

Slice 070 added the operation-level kwarg `seed_claim_status: str =
"draft"`. The effective lifecycle status of every deterministic seed
claim is resolved as: per-candidate `SeedClaim.status` if explicitly
supplied, otherwise `seed_claim_status`. Model-backed candidates
keep the status the parser set. The kwarg must be a value in
`CLAIM_STATUSES`; anything else refuses the batch atomically before
any harness call (no `SourceDocument` enters `LLMInvoke`, no
claim/entity/page/fingerprint write occurs). Curated seed scripts
can opt in to `"reviewed"` directly via the library; the first-class
`llloom seed --status reviewed --manifest ...` CLI surface is
deferred to Slice 075. The default `"draft"` preserves the historical
seed behavior. `SeedClaim.status`'s default changed from the string
`"draft"` to `None` so the two signals can be distinguished — every
existing caller that constructs `SeedClaim(...)` without naming
`status=` keeps landing at `"draft"` because that is the
`seed_claim_status` default.

```python
from llloom.claims.models import Locator
from llloom.ops.ingest import SeedClaim, ingest

ingest(
    workspace,
    path_to_raw_source,
    source_id="src.example",
    source_class="markdown_prose",
    seed_claims=[
        SeedClaim(
            entity_id="concept.example",
            entity_type="concept",
            display_name="Example",
            claim_id="c_0001",
            claim_kind="definition",
            claim_text="...",
            locator=Locator(
                locator_type="markdown_prose_v1",
                heading_path=["Methods"],
                paragraph_index=1,
                sentence_start=1,
                sentence_end=1,
            ),
            render_target=("concept/example", "claim_block.concept.example"),
        ),
    ],
)
```

## LLM harness API

```python
from llloom.llm import (
    ALLOWED_OPERATIONS,
    ALLOWED_READ_CLASSES,
    ALLOWED_WRITE_KINDS,
    LLMInvoke,
    ModelOutputError,
    NullModel,
    RawCandidate,
    SourceDocument,
    ClaimRecord,
    SourceSpan,
    ClaimBlockRegion,
    SchemaDocument,
    StructureItemContext,
    WriteTarget,
    HarnessRefusal,
    parse_claim_extraction_output,
)
```

`StructureItemContext` is a frozen dataclass carrying metadata-only
fields (`source_id`, `source_class`, `language`, `kind`, `name`,
`symbol_path`, `report_path`) plus a deterministic `content_hash`.
It is the typed-input class for **explicit, caller-selected
structure context** on `claim_extract` /
`claim_extract_and_view_render` ingest of a narrative source. The
harness allows it only for `operation_kind == "ingest"`; render,
query, and lint refuse it. No raw code text, comments, docstrings,
scalar values, or `code_v1` excerpt bytes are carried.

`LLMInvoke(model=...)` accepts any `ModelBackend` (a protocol with an
``identifier`` attribute and a ``generate(prompt) -> str`` method). The
default is `NullModel` (returns empty output). The harness has no file
or network access; callers must construct typed inputs themselves.

`ALLOWED_READ_CLASSES` is the per-operation typed-input allow list and
`ALLOWED_WRITE_KINDS` is the per-operation write-target allow list.
Callers can introspect these mappings to know what is legal for each
operation; the harness enforces both on every `invoke` call.

Every `LLMInvoke.invoke` call returns ``(output, log)`` where ``log``
is an ``InvocationLog``. Mutating operations persist a summary of the
log into the operation journal entry under the ``invocation_logs``
field. The summary contains the typed input class name and content
hash for every input -- never raw source text.

### Model output contract for `claim_extract` ingestion

`parse_claim_extraction_output(output_text) -> list[RawCandidate]`
parses model output per the strict YAML contract documented in
`04_specification/component_contracts.md` "Model output contract for
`claim_extract` ingestion". Empty/whitespace-only input returns the
empty list; any structural problem raises `ModelOutputError`. The
ingest pipeline converts each `RawCandidate` into the existing
`SeedClaim` shape and runs them through the same provenance verifier
as explicit caller-supplied seeds. Persistence is **batch atomic**:
a single failure refuses the whole batch.

`IngestResult.extraction_errors` records per-candidate failure notes
when extraction refuses; on a refused batch `refusal_reason` is set
and `succeeded` is False.

The strict YAML parser is source-class-aware via an
`allowed_locator_types: set[str]` keyword:
`parse_claim_extraction_output(output_text, *,
allowed_locator_types=...)`. The ingest path picks the set from the
schema source-class locator. Narrative source classes admit only
their narrative locator type; an explicit **code-backed**
`claim_extract` ingest (source class whose locator is `code_v1`)
admits only `code_v1`. Any candidate whose `locator.locator_type` is
not in the allowed set raises `ModelOutputError` (batch-atomic
refusal). Legacy callers that omit the keyword fall back to
narrative-only behavior (`{markdown_prose_v1, legal_act_v1}`), so
the prior contract remains the default.

## Direct `code_v1` claims on code-backed `claim_extract`

A source class whose locator is `code_v1` and whose policy is
`claim_extract` admits a model-emitted `code_v1` claim, subject to
**combined deterministic validation**. After parsing, `ops.ingest`
re-runs the deterministic structure extractor on the raw source and
admits a candidate's `code_v1` locator iff it matches one of two
shapes (compared on the six addressable keys `locator_type`, `path`,
`start_line`, `start_col`, `end_line`, `end_col`):

- a **declaration-level span** — a structure-item locator exactly
  (class / function / method / type / interface / trait / enum /
  struct definitions); or
- an **attached explanation span** — a contiguous line-comment
  block immediately above one of those declarations with no
  blank line between, or a Python triple-quoted docstring on the
  line immediately below a class / function / async-function
  declaration. Both shapes are enumerated deterministically from
  the current raw source text. Explanation locators cover whole
  lines (`start_col == 1`, `end_col` = the last covered line's
  full length).

Behavior:

- detached / free-floating comments, arbitrary code-body spans,
  and fabricated line ranges refuse the whole batch
- the verifier still re-resolves and re-hashes the exact span on
  every admitted candidate; the `code_v1` excerpt is preserved
  byte-for-byte by `normalize_excerpt`
- a code-backed source class configured for
  `claim_extract_and_view_render` is now supported. The same
  combined `code_v1` validator runs first; only after a
  successful batch persistence does the existing render path
  fire, using the existing variant-(B) page contract (renders
  only inside the claim-block region, commentary survives
  byte-for-byte, malformed / missing page markers still fail
  hard). The render path reads authoritative claim state — not
  raw source text. Invalid code-backed batches still refuse
  with no claim or page mutation
- the narrative `claim_extract` path is unchanged: it still refuses
  `code_v1` because the schema source class's locator names a
  narrative type, so the allowed set never contains `code_v1`

## Metadata-only structure context on `claim_extract`

`ingest(..., structure_source_ids=[...])` (CLI: repeatable
`--structure-source <source_id>`) lets a caller pass explicit
registered structure-source ids whose on-disk reports under
`state/structure/<source_id>.yaml` should be loaded as metadata-only
`StructureItemContext` blocks on the `LLMInvoke` prompt for
`claim_extract` and `claim_extract_and_view_render`. Only structure
metadata reaches the model (source id, source class, language, item
kind, name, symbol path, report path); scalar values, comments,
docstrings, full source lines, and code bodies are never carried.

Behavior:

- only `claim_extract` and `claim_extract_and_view_render` consume
  the list; `index_only`, `structure_extract`, and `deny` ignore it
  and still return before any `LLMInvoke` call
- persisted claims must still ground in the narrative source under
  ingest (locator `markdown_prose_v1` or `legal_act_v1`); `code_v1`
  on this path is refused by the strict YAML parser
- a requested structure source that is missing from the registry,
  retracted, or whose report file is absent, malformed, or stale
  (mismatched `version`, `source_id`, `source_class`, or
  `content_hash`) refuses the ingest cleanly with no partial
  writes — silent omission would mislead the caller who explicitly
  requested the context
- the invocation log summary records one entry per
  `StructureItemContext` carrying class / id (`<source_id>:<symbol_path>`) /
  content hash only, never any code or scalar text

`StructureItemContext` is opt-in. Default ingest behavior is
unchanged.

## CLI

Installed via `pyproject.toml` as a console script:

```text
llloom init
llloom status
llloom ingest <path-or-source-id> [--source-id ID] [--source-class CLASS] [--no-render] [--model-provider openai|anthropic --model <model-id> [--model-timeout <seconds>]] [--structure-source <source_id> ...]
llloom verify [target]
llloom render [page:<page_id>] [--dry-run] [--list-targets]
llloom seed apply <manifest.yaml> [--dry-run] [--no-render] [--status <status>]
llloom query "<question>"
llloom lint [--generated-canary]
llloom reconcile
llloom unlock <target> --reason "<reason>"        # bare unlock window (journal-only)
llloom unlock --clear-stale --reason "<reason>"   # guarded stale-lock clear
llloom unlock --dead-owner --reason "<reason>"    # guarded local same-host dead-owner clear (Slice 086)
llloom promote <target> --to reviewed|validated|superseded|archived
llloom retract <source_id> [--reason "<reason>"]
llloom rebuild <target>     # render_fingerprints|health_report|index|log|search|graph
llloom list_merge_proposals
llloom review-alias <proposal_id> --decision approve|reject [--notes ...]
llloom merge-alias <proposal_id>
llloom reject-alias <proposal_id> [--notes ...]
llloom page create <page_id> [--page-class entity|concept|synthesis|navigation] [--title TITLE]
```

All commands accept `--root <path>` (defaults to `.`). Output is JSON
on stdout so the CLI composes cleanly with other tools. Non-zero exit
codes indicate refusal or lint failure.

## Stability guarantees

- `Workspace` path attributes are stable: attribute names match the
  frozen `04_specification/storage_and_state_model.md` layout.
- Operation result dataclass field names are stable for the first
  slice. Future additions will be additive.
- The `LLMInvoke` typed-input classes are stable. New classes will only
  be added if the operation matrix expands.

## Search sidecar helpers

```python
from llloom.state import (
    SearchHit,
    SearchSidecarError,
    build_search_sidecar,
    search_candidates,
    sidecar_exists,
)
```

`build_search_sidecar(workspace)` (also reachable via
`rebuild(workspace, target="search")`) creates or replaces the
SQLite FTS5 sidecar at ``workspace.search_db`` (preferred path
``state/search/search.sqlite``). The sidecar is derived state only:
it may be deleted without data loss. `rebuild search` returns a
dict ``{"target": "search", "index_path", "claim_rows",
"source_rows"}``.

`search_candidates(workspace, query_text, *, limit=50)` returns
``list[SearchHit]`` — candidate ids for claims, ``index_only``
sources, or structure-report items. Callers must rehydrate and
revalidate each hit against canonical records (or, for structure
items, against the on-disk report under ``state/structure/``)
before trusting it; this is exactly what ``query`` does
internally. If the sidecar is absent, `search_candidates`
returns the empty list.

`SearchHit` exposes:

- `doc_type: str` — `"claim" | "index_only_source" | "structure_item"`
- `entity_id: str | None`, `claim_id: str | None`,
  `source_id: str | None`
- `rank: float`
- `structure_kind: str | None`, `structure_name: str | None`,
  `structure_symbol_path: str | None`,
  `structure_language: str | None`,
  `structure_report_path: str | None` — non-None only for
  `doc_type == "structure_item"` rows

`rebuild(workspace, target="search")` returns
``{"target": "search", "index_path", "claim_rows",
"source_rows", "structure_rows"}``. The `structure_rows` count
is the number of structure-report items indexed this pass;
malformed or stale reports are silently skipped and contribute
zero rows.

Structure-item rows carry **metadata only**: source id, source
class, language, item kind, name, symbol path, and the
workspace-relative report path. Scalar YAML values, comments,
docstrings, full source lines, code bodies, raw source bodies,
rendered page prose, commentary, journal text,
locator-resolved excerpts, and model output are never indexed.

`SearchSidecarError` is raised when SQLite FTS5 is not available
in the local `sqlite3` build. The sidecar never falls back to a
weaker index.

## Graph sidecar helpers

```python
from llloom.state import (
    GraphEdge,
    GraphSidecarError,
    StructureGraphEdge,
    build_graph_sidecar,
    graph_neighbors,
    graph_sidecar_exists,
    structure_graph_neighbors,
)
```

`build_graph_sidecar(workspace)` (also reachable via
`rebuild(workspace, target="graph")`) creates or replaces the
SQLite graph sidecar at ``workspace.graph_db`` (preferred path
``state/graph/graph.sqlite``). The sidecar is derived state only
and may be deleted without data loss. `rebuild graph` returns
``{"target": "graph", "index_path", "edge_rows",
"structure_edge_rows"}``. The `edge_rows` count is the
claim-relation edge count (existing behavior); the
`structure_edge_rows` count is the number of direct parent/child
containment edges indexed across all derived structure reports
(additive).

`graph_neighbors(workspace, *, claim_id, direction="both",
relation_types=None, include_inactive=False, limit=50)` returns
``list[GraphEdge]``. Direction is one of ``"in" | "out" | "both"``;
other values raise `GraphSidecarError`. If the sidecar exists it is
used to narrow candidate relation ids, but every returned edge is
rehydrated from canonical entity YAML and revalidated — the owning
entity must still contain the relation, both endpoint claims must
exist, and both endpoints and the relation must be active unless
``include_inactive=True``. If the sidecar is absent,
`graph_neighbors` falls back to a full canonical scan and still
returns correct edges.

`GraphEdge` exposes `relation_id`, `source_entity_id`,
`source_claim_id`, `relation_type`, `target_entity_id`,
`target_claim_id`, and `status`. Edges are always rehydrated from
canonical YAML before emission.

A non-positive ``limit`` (``limit <= 0``) returns the empty list
without touching the sidecar or walking canonical YAML. This
matches common Python slicing expectations and keeps the helper
easy to use from agents and composed callers.

`structure_graph_neighbors(workspace, *, source_id, symbol_path,
direction="both", limit=50)` returns ``list[StructureGraphEdge]``
— the structure analogue of `graph_neighbors` over direct
parent/child containment edges from the on-disk structure report
at ``state/structure/<source_id>.yaml``. Direction is one of
``"in" | "out" | "both"``; other values raise
`GraphSidecarError`. If the sidecar exists it narrows candidate
``(parent_symbol_path, child_symbol_path)`` pairs, but every
returned edge is rebuilt from the current report and revalidated
against the current source registry record (``status`` !=
``retracted``, matching ``source_class``, matching
``content_hash``). If the sidecar is absent,
`structure_graph_neighbors` walks the current report directly and
still returns correct edges. ``limit <= 0`` returns ``[]`` without
touching the sidecar or the report. Output preserves the report's
deterministic item order so callers see the same edges across
runs regardless of SQLite row order. The helper performs **direct
containment only**; ancestor shortcuts, cross-source edges, and
multi-hop traversal are not in scope.

`StructureGraphEdge` exposes `source_id`, `source_class`,
`language`, `parent_symbol_path`, `child_symbol_path`,
`child_kind`, `child_name`, and `report_path`. Edges carry
**metadata only** (no scalar values, no comments, no docstrings,
no code bodies, no raw source text, no locator-resolved excerpts,
no rendered prose, no model output) and are always rehydrated
from the on-disk report before emission.

## Optional provider adapters

```python
from llloom.llm import OpenAIBackendError, OpenAIModelBackend
# or, to avoid touching the re-export:
from llloom.llm.openai_backend import OpenAIBackendError, OpenAIModelBackend
```

`OpenAIModelBackend` implements the existing `ModelBackend`
protocol. Installed via the optional extra:

```bash
pip install "llloom[openai]"
```

Constructor fields:

- `model: str` (default: `"gpt-5.4"` — a default, not a guarantee
  of currency or superiority; callers should pass an explicit
  `model` string for pinned behavior. `gpt-5.4-mini` and
  `gpt-5.4-nano` are common cost/latency-sensitive alternatives.)
- `api_key: str | None = None` (defaults to the SDK's environment
  lookup, typically `OPENAI_API_KEY`)
- `base_url: str | None = None`
- `timeout: float | None = None`
- `reasoning_effort: str | None = None`
- `max_output_tokens: int | None = None`

Behavior:

- `identifier == f"openai/{model}"` is reported by the harness in
  the invocation log; the log still carries only typed-input class
  + content hash for every input, never raw source text.
- `generate(prompt)` calls the OpenAI Responses API
  (`client.responses.create(...)`) with the deterministic prompt
  the harness assembled. The adapter prefers
  `response.output_text` and raises `OpenAIBackendError` when no
  text output can be recovered. The adapter never silently returns
  an empty string.
- The adapter does not enable tool calling, web search, file
  search, code interpreter, background mode, or hosted retrieval.
  The prompt is the only context the model sees.
- The adapter does not touch workspace state, journals, pages,
  commentary, spine prose, the search sidecar, or the graph
  sidecar.

`OpenAIBackendError` is raised when the optional dependency is not
installed or when the SDK returns a response with no extractable
text. Importing `llloom` or `llloom.llm` must not fail when the
SDK is absent; only constructing a client inside `generate(...)`
imports the SDK.

The strict YAML output contract documented in
`04_specification/component_contracts.md` §"Model output contract"
still applies; the provider adapter does not relax parsing. A
malformed response is a batch-atomic refusal with `refusal_reason`
set and no partial persistence.

### Anthropic Claude adapter

```python
from llloom.llm import AnthropicBackendError, AnthropicModelBackend
# or, to avoid touching the re-export:
from llloom.llm.anthropic_backend import AnthropicBackendError, AnthropicModelBackend
```

`AnthropicModelBackend` implements the existing `ModelBackend`
protocol. Installed via the optional extra:

```bash
pip install "llloom[anthropic]"
```

Constructor fields:

- `model: str` (required — no default; pass an explicit Anthropic
  model id, e.g. `claude-sonnet-4-5-20250929`)
- `api_key: str | None = None` (defaults to the SDK's environment
  lookup, typically `ANTHROPIC_API_KEY`)
- `timeout: float | None = None`
- `max_output_tokens: int = 4096` (Anthropic's Messages API
  requires `max_tokens`; the adapter sets a conservative default)
- `temperature: float | None = None`
- `system_prompt: str` (a narrow sibling of the OpenAI
  instructions string; YAML-only, no fences, no commentary, never
  invent source text, `claims: []` on doubt, never emit
  `code_v1`)

Behavior:

- `identifier == f"anthropic/{model}"` is reported by the harness
  in the invocation log; the log still carries only typed-input
  class + content hash for every input, never raw source text.
- `generate(prompt)` calls the Anthropic Messages API
  (`client.messages.create(model=..., max_tokens=...,
  system=..., messages=[{"role": "user", "content": prompt}])`).
  The adapter is single-turn: no streaming, no message batches,
  no tools, no tool use, no web search, no computer use, no
  files / hosted retrieval, no background mode, no multi-turn
  conversation state. The prompt is the only context the model
  sees.
- The adapter walks the response's `content` blocks and
  concatenates `text` from every block whose `type == "text"` in
  the order the SDK returned them; non-text blocks (tool use,
  images, thinking, etc.) are silently ignored. The adapter
  raises `AnthropicBackendError` if no text output can be
  recovered; it never silently returns an empty string.
- The adapter does not touch workspace state, journals, pages,
  commentary, spine prose, the search sidecar, or the graph
  sidecar.

`AnthropicBackendError` is raised when the optional dependency is
not installed or when the SDK returns a response with no
extractable text. Importing `llloom` or `llloom.llm` must not
fail when the SDK is absent; only constructing a client inside
`generate(...)` imports the SDK.

The strict YAML output contract applies to both providers
identically; the Anthropic adapter does not relax parsing.

CLI shape (default behavior unchanged; provider flags are opt-in):

```text
llloom ingest <path> [--no-render] \
    [--model-provider openai --model <model-id> [--model-timeout <seconds>]]
llloom ingest <path> [--no-render] \
    [--model-provider anthropic --model <model-id> [--model-timeout <seconds>]]
```

`--model-provider openai` without the `llloom[openai]` extra
installed exits non-zero with a helpful error that names the
install extra; no JSON ingest result is emitted. The same
behavior applies for `--model-provider anthropic` and
`llloom[anthropic]`.

## Structured-source ingest

```python
from llloom.structured import (
    StructureExtractError,
    StructureItem,
    StructureReport,
    extract_structure,
    write_structure_report,
    SUPPORTED_SOURCE_CLASSES,
)
```

`extract_structure(source_text, *, source_id, source_class,
locator_type, raw_path, content_hash) -> StructureReport` emits a
deterministic derived report for a structured source.
`write_structure_report(workspace, report) -> Path` atomically
persists the report under `state/structure/<source_id>.yaml`.

Supported source classes:

- `structured_yaml` — works in the base install via PyYAML
- `raw_evidence` (Slice 083) — neutral starter class for
  unsupported or intentionally unstructured UTF-8 evidence.
  Maps to the existing `index_only` ingest policy: registers
  the source + hashes it + supports exact deterministic
  retrieval via `query(...)` (the same path the pre-existing
  `index_only` retrieval uses); creates no claims, no pages,
  no structure reports; never invokes `LLMInvoke`. Reuses the
  `markdown_prose_v1` locator shape internally only — no new
  locator type is introduced. Use it for `.java` (before the
  Java structure-extraction extra is installed), `.kt`,
  `.proto`, or any UTF-8 evidence llloom does not yet parse.
  Not a substitute for `code` when structured extraction is
  available.
- `code` — code-structure extraction across Python (`.py`), Go
  (`.go`), Rust (`.rs`), TypeScript (`.ts`), C# (`.cs`), and
  Java (`.java`); requires the optional extra
  `pip install "llloom[structured]"` (tree-sitter +
  tree-sitter-python + tree-sitter-go + tree-sitter-rust +
  tree-sitter-typescript + tree-sitter-c-sharp +
  tree-sitter-java). Each language package is lazy-imported
  inside the per-language loader, so importing `llloom` or
  `llloom.structured` never requires the extra; missing-extra
  raises `StructureExtractError` whose message names
  `llloom[structured]`. Unsupported `code` file suffixes
  (`.tsx`, `.js`, `.kt`, etc.) refuse with a clear error that
  names the supported suffix set and the install extra.

Suggested per-language `kind` values (closed set in this slice;
no decorators, fields, locals, imports, references, or
comments):

- Python — `class`, `function`, `async_function`
- Go — `function`, `method`, `type`
- Rust — `function`, `struct`, `enum`, `trait`, `method`
- TypeScript — `function`, `class`, `interface`, `type_alias`,
  `method`
- C# — `class`, `interface`, `struct`, `enum`, `method`,
  plus `unity_component` (Unity bridge v1): a
  `class_declaration` that directly inherits from
  `MonoBehaviour` (or a qualified name ending in
  `.MonoBehaviour`) is re-tagged from `class` to
  `unity_component`. Method kinds inside such a class remain
  `method` and `symbol_path` qualification is unchanged. The
  detection is direct-base only — no transitive inheritance,
  no alias / using-graph resolution. `ScriptableObject`,
  lifecycle interpretation, and other Unity engine semantics
  remain deferred
- Java (Slice 082) — `class`, `interface`, `enum`, `record`,
  `method`, `constructor`, `field`. Constructors are emitted
  with `symbol_path == "<Class>.<Class>"` so the
  invariant-on-name path qualifies cleanly under the
  enclosing class. Fields are emitted one per
  `variable_declarator` under a `field_declaration` (the
  first declarator wins in V1 — multi-declarator forms like
  `int a, b;` emit only `a`). Annotations, modifiers,
  imports, throws clauses, parameter names, statement
  bodies, scalar literals, and comments are deliberately out
  of scope. No call graph, no type resolution, no
  Maven/Gradle awareness, no transitive inheritance, no
  framework-tagging classifier (no Java analogue of the C#
  Unity bridge in V1)

Method `symbol_path` values are consistently dot-qualified by the
enclosing receiver / impl target / class — e.g. `Store.Save`
(Go), `Store.save` (Rust impl method), `Store.save` (TypeScript
class method), `Store.Save` (C# class method), `Project.evaluate`
(Java class method). Java constructors qualify as
`Project.Project`; Java fields qualify as `Project.maxsize`.

Every emitted `StructureItem` and its serialized form now carries
a generic `tags: tuple[str, ...]` channel (`tags: list[str]` in
the YAML report). Tags are lowercase ASCII `prefix:value`
strings and default to `()` / `[]`. The first shipped classifier
attaches `("framework:unity", "role:component")` to direct C#
`MonoBehaviour` subclasses (whose `kind` remains
`unity_component`); every other extractor path produces
empty-tagged items. Tags never replace `kind`, never drive
verifier, render, or claim semantics, and are **not**
user-configurable in this slice — they are a small, additive
metadata hook that future framework classifiers can reuse
without binding `llloom` core to any specific framework.
`StructureItem.from_mapping(...)` normalizes legacy
`structure_report_v1` items written before this slice (no `tags`
key) to `tags=()`, preserving rehydration compatibility.

Reports are **non-canonical**: deletable without data loss,
rebuildable by re-ingest. Reports contain structure only (key
paths, symbols, kinds, `code_v1` locators, generic framework
tags). Scalar values, comments, docstrings, full source lines,
and code bodies are **not** stored.

The `code_v1` locator shape (1-based inclusive line/col; whitespace
preserved on normalize) is resolved by
`llloom.claims.locators.resolve_span`.

`IngestResult.structure_reports: list[str]` carries the
workspace-relative POSIX path of any structure report a
`structure_extract` ingest wrote. Empty on every other policy.

Safety:

- `structure_extract` never invokes `LLMInvoke`.
- No `SourceDocument` is constructed on this path.
- Providers must not emit `code_v1` locators on the
  `claim_extract` path; the OpenAI backend's instructions
  continue to forbid `code_v1` explicitly.
- Malformed sources refuse cleanly with no partial report.

## MCP server (first slice)

```bash
pip install "llloom[mcp]"
llloom-mcp --root path/to/workspace
```

`llloom-mcp` is a **separate console script**, not a new `llloom`
operation verb. The core `llloom` CLI verb set is unchanged by
this surface; see the §CLI section above for the current verb
list and the `tests/contract/test_prepare_pdf_cli.py`
`EXPECTED_VERBS` guard for the authoritative count. The MCP server
binds one workspace root at process startup; tool calls cannot
switch roots.

Transport: local stdio only. HTTP / SSE / WebSocket transports,
background mode, editor plugins, and hosted service are deferred.

Tool surface (first slice — read-only and diagnostic only):

- `llloom_status()` → JSON `StatusResult`
- `llloom_query(question: str)` → JSON `QueryResult`; never
  invokes `LLMInvoke`
- `llloom_verify(target: str | None = None)` → JSON
  `VerifyResult` (includes structured `mismatches`)
- `llloom_lint(generated_canary: bool = False)` → JSON
  `LintResult`; `canary_hits` behaviour identical to CLI / library
- `llloom_graph_neighbors(claim_id, direction="both",
  relation_types=None, include_inactive=False, limit=50)` →
  JSON `list[GraphEdge]`; `limit <= 0` returns `[]` and unknown
  directions raise a tool error carrying the
  `GraphSidecarError` message verbatim
- `llloom_list_merge_proposals()` → JSON `list[MergeProposalSummary]`

Mutating tools are **deliberately not exposed** in this slice:
no `ingest`, `retract`, `promote`, `review-alias`, `merge-alias`,
`reject-alias`, `unlock`, `reconcile`, or `rebuild` MCP tool is
registered. Defence-in-depth: the server refuses to start if any
forbidden tool name is observed in its registry.

All tool results are produced by a local `to_jsonable(...)`
helper that walks dataclasses via `dataclasses.asdict`,
normalizes `Path` to `str`, and leaves lists / dicts / primitives
as-is. Existing result dataclass shapes are preserved; no new
result dataclass is introduced by this slice.

If the optional extra is missing, `llloom-mcp` exits with code 2
and a stderr message naming `llloom[mcp]`; no tool is served.

## What is NOT in the API yet

- mutating MCP tools
- HTTP / SSE / WebSocket MCP transport, hosted service, background
  watch mode, editor plugins
- broad-language tree-sitter support beyond the first
  structured-source set (YAML in the base install; Python, Go,
  Rust, TypeScript, C#, and Java via `llloom[structured]`); Unity
  engine semantics beyond the Unity bridge v1 classification
  (direct `MonoBehaviour` subclasses surface as
  `unity_component`): `ScriptableObject` classification,
  lifecycle interpretation, `.unity` / `.prefab` / `.asmdef`
  parsing, serialized-field semantics, asset reference
  reasoning, editor / IDE integration; detached or
  free-floating comment / docstring claims,
  arbitrary
  code-body-span claims, MCP structured-ingest tools
  (search-sidecar indexing of structure-report metadata,
  `query` surfacing of rehydrated `StructureItemHit` records,
  graph-sidecar indexing of direct parent/child containment
  with `structure_graph_neighbors(...)`, metadata-only
  structure context on narrative `claim_extract` via
  `--structure-source` / `structure_source_ids=[...]`, direct
  `code_v1` claims on code-backed `claim_extract` and
  `claim_extract_and_view_render` for declaration-level spans
  plus attached explanation spans (leading line-comment block
  / Python docstring) are implemented)
- graph visualization and multi-hop query expansion
- vector / semantic re-ranking
- Gemini, local-model, Ollama, and LiteLLM adapters (Anthropic
  is implemented behind the `llloom[anthropic]` optional extra)
- multi-provider orchestration / routing / retry queues
- streaming model APIs, tool calling, and hosted retrieval
- `watch`, `serve`, `autowiki`, `research` verbs

These are deferred per `04_specification/package_boundary.md`.

