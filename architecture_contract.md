# Package Architecture Contract

Concrete module layout as implemented in `src/llloom/`.

## Modules

### `workspace`
Path resolution and layout validation. Owns:

- `Workspace` dataclass with typed path attributes
- `Workspace.load(root)` and `Workspace.init(root)`
- `REQUIRED_DIRS` / `REQUIRED_SCHEMA_FILES` constants
- `WorkspaceError`

Out of scope: any read/write of content. The workspace object is a
navigation contract, not a mutation contract.

### `schema`
Policy resolution. Owns:

- `Schema`, `SourceClass`, `PageClass`
- `load_schema(workspace)`
- `INGEST_POLICIES` (the five frozen policy names)
- `SchemaError`

Validates locator types up front and enforces that
`ingest_policies.yaml` never references an undefined source class.

### `sources`
Source registration and hashing. Owns:

- `SourceRecord`, `SourceRegistry`, `SourceRegistryError`
- atomic YAML persistence (temp-file-and-rename)
- hash-based immutability check (`register` refuses modified evidence)

### `claims`
Authoritative machine-usable interpretation. Owns:

- `Assertion`, `Evidence`, `Locator`, `RenderTarget`, `Relation`,
  `Alias`, `EntityContainer`, `MergeProposal`
- `ClaimStore` (one YAML file per entity under `claims/entities/`,
  separate `claims/merge_proposals/` for review artifacts)
- `locators.resolve_span`, `locators.normalize_excerpt`
- `verifier.verify_assertion`, `verifier.verify_evidence`,
  `verifier.compute_excerpt_hash`

The excerpt hash is mandatory on every evidence entry; the verifier
re-resolves the locator, re-normalizes, and re-hashes.

### `pages`
Variant-(B) page layer. Owns:

- `regions.parse_page`, `regions.replace_claim_block`, `ParsedPage`
- HTML-comment markers: exactly one claim-block pair and one commentary
  pair per page; malformed or duplicate markers are hard failures
- `render.render_page_file` (deterministic template expansion; does NOT
  invoke an LLM for render in the first slice)
- `render.compute_render_fingerprint`
- `render.resolve_page_path`

The renderer writes only inside the claim-block region; commentary
survives byte-for-byte.

### `ops`
User-facing operations. Each verb is a separate module and returns a
dataclass result:

- `ingest`, `verify`, `render`, `query`, `lint`, `reconcile`, `unlock`,
  `promote`, `retract`, `rebuild`, `status`
- `alias.list_merge_proposals`, `alias.review_alias`,
  `alias.merge_alias`, `alias.reject_alias`
- `page.create_page` (Slice 084) — deterministic page-stub
  creator. Writes one valid variant-(B) page under
  `pages/<class_dir>/<tail>.md`, derives marker ids from the
  normalized `page_id`, refuses pre-existing targets, never
  creates claims, never invokes a model, never updates render
  fingerprints. Runs under the existing
  `_context.operation(...)` lock + journal contract
- `_context.operation(...)` â€” context manager that acquires the
  workspace lock and opens the operation journal
- `prepare_pdf` — optional PDF working-text on-ramp; reuses the
  same `_context.operation(...)` lock + journal pattern; never
  touches `claims/`, `pages/`, or `state/source_registry/`; never
  invokes a model. Output lives under
  `raw/derived/pdf/<prep_id>/`.

### `pdf_prep`
Optional first-party PDF working-text prep. The only place that
imports `docling`, and only inside the
`pdf_prep.docling.convert_with_docling` function so the base
install does not require the optional extra. The op layer talks
to the adapter through the `PdfPrepAdapter` Protocol; tests
inject fake adapters. The `pdf_prep.manifest` module owns the
`pdf_prep_manifest_v1` shape with its reserved component slots
(`pymupdf` / `grobid` / `pdfplumber` / `nougat`) so a future
companion producer can fill them in without schema migration.
Verifier and locator semantics are unchanged: the selected
ingest artifact is ordinary Markdown ingested through the normal
`llloom ingest` path.

### `llm`
Single invocation harness. Owns:

- typed input classes: `SourceDocument`, `ClaimRecord`, `SourceSpan`,
  `ClaimBlockRegion`, `SchemaDocument`, `StructureItemContext`
  (metadata-only summary of one structure-report item: `source_id`,
  `source_class`, `language`, `kind`, `name`, `symbol_path`,
  `report_path`; allowed only on `ingest`; never carries raw code
  text, scalar values, comments, or docstrings), `WriteTarget`
- `LLMInvoke.invoke(...)` â€” the only authorized gateway for
  LLM-invoking operations
- `ALLOWED_READ_CLASSES[op]` â€” frozen per-operation allow list of typed
  input classes that may appear at all
- `ALLOWED_WRITE_KINDS[op]` â€” frozen per-operation allow list of write
  target kinds
- `NullModel` (default backend) and `ModelBackend` protocol
- `HarnessRefusal`, `InvocationLog`
- optional provider adapter `OpenAIModelBackend` in
  `llloom.llm.openai_backend` (installed via the `llloom[openai]`
  extra). Lazy imports the SDK inside `generate(prompt)` so
  importing `llloom.llm` never requires the optional dependency.
  The adapter receives only the deterministic prompt the harness
  assembled; it does not touch workspace state, pages, commentary,
  spine prose, journals, the search sidecar, or the graph sidecar.
  Output text is parsed by the existing strict YAML parser
  (`parse_claim_extraction_output`); batch atomicity is preserved.
- optional provider adapter `AnthropicModelBackend` in
  `llloom.llm.anthropic_backend` (installed via the
  `llloom[anthropic]` extra). Sibling of the OpenAI adapter:
  lazy-imports the SDK inside `generate(prompt)`, calls
  `client.messages.create(model=..., max_tokens=...,
  system=..., messages=[{"role": "user", "content": prompt}])`
  as a single-turn text adapter (no streaming, no batches, no
  tools, no web search / files / hosted retrieval, no
  multi-turn state). Walks `message.content` and concatenates
  `text` from every `type == "text"` block in SDK order;
  non-text blocks are silently ignored; missing text raises
  `AnthropicBackendError`. Output is parsed by the same strict
  YAML parser; the system prompt is a narrow sibling of the
  OpenAI instructions (YAML-only, no fences, never invent
  source text, `claims: []` on doubt). Both providers'
  instructions are now **source-class-aware**: narrative source
  classes still forbid `code_v1`, and code-backed `claim_extract`
  / `claim_extract_and_view_render` ingests admit `code_v1` for
  either declaration-level spans or attached explanation spans
  (leading line-comment block above a declaration, or a Python
  triple-quoted docstring on the line immediately below a class
  / function / async-function declaration). Detached comments
  and arbitrary code-body spans are still forbidden. The strict
  YAML parser accepts an `allowed_locator_types` keyword that
  `ops.ingest` picks from the schema source-class locator.
  Code-backed `claim_extract_and_view_render` reuses the
  existing variant-(B) page renderer (renders only inside the
  claim-block region, commentary survives byte-for-byte) after
  successful claim persistence; no raw source body enters the
  render step.

No filesystem or network access. Excluded content classes
(commentary, spine prose, `index_only` source bodies) have no typed
input class at all â€” unreachable by construction. Inputs that DO have
a typed class but are inappropriate for the operation (e.g. raw
`SourceDocument` reaching `render` or `query`) are refused by the
read-class allow list.

### `state`
Workspace-scoped state. Owns:

- `WorkspaceLock` with `is_timed_out` (heartbeat-only) and
  `is_stale_recoverable(lock, journal=...)` (journal-backed). Slice
  085 added four optional owner-process metadata fields on the
  `Lock` dataclass (`owner_pid`, `owner_hostname`, `owner_cwd`,
  `owner_command`) plus a conservative
  `local_owner_pid_state(lock, *, current_hostname=None)` helper
  returning `"alive"` / `"dead"` / `"unknown"`. Metadata is
  diagnostic only — `is_stale_recoverable` is unchanged. Slice 086
  adds `unlock(..., dead_owner=True, reason="...")` — a guarded
  local same-host operator escape hatch that clears the workspace
  lock only when every local predicate holds (same-host owner
  metadata, `local_owner_pid_state == "dead"`, lock not yet timed
  out, matching journal entry exists and is `in_progress` with no
  `completed_at`, identical pre-clear re-read). Mutually exclusive
  with `clear_stale`. Audit op_kind:
  `unlock_clear_dead_owner`. `is_stale_recoverable` and
  `reconcile` remain byte-identical.
- `OperationJournal` (YAML per op under `state/journals/<op_id>.yaml`),
  including a persisted `invocation_logs` field that records every
  `LLMInvoke` call summary (typed input class + content hash, never
  raw text)
- `FingerprintStore` for `state/render_fingerprints.yaml`
- `search` module: the hybrid search sidecar at
  `state/search/search.sqlite` (SQLite FTS5). `build_search_sidecar`
  creates a new sidecar into a temp file and atomically replaces
  the old one; `search_candidates` returns `SearchHit` rows that
  `ops.query` rehydrates from canonical claim YAML, raw source
  files, or the on-disk structure report under
  `state/structure/<source_id>.yaml` before emitting citations,
  verbatim spans, or `StructureItemHit` records. The sidecar is
  derived state only and may be deleted without data loss.
  Retracted / archived / stale / superseded claims and retracted
  `index_only` sources are excluded at build time and filtered
  again at query time; structure reports whose source is
  missing, retracted, or whose registered source class / content
  hash no longer matches contribute zero rows; commentary, spine
  prose, rendered page prose, journals, health reports,
  merge-proposal prose, raw source bodies on the structure-report
  path, and model output are never indexed. Structure-item rows
  store metadata only (`structure_kind`, `structure_name`,
  `structure_symbol_path`, `structure_language`,
  `structure_report_path`).
- `graph` module: the graph sidecar at `state/graph/graph.sqlite`
  (plain SQLite). Carries two derived tables. The `edges` table
  indexes claim-relation edges: `build_graph_sidecar` writes
  canonical active relation records into a temp file and replaces
  the old database directly; `graph_neighbors(workspace, *,
  claim_id, direction, relation_types, include_inactive, limit)`
  uses the sidecar only to narrow candidate relation ids and
  rehydrates every `GraphEdge` from canonical entity YAML before
  emission, revalidating endpoint existence and active status.
  The `structure_edges` table indexes direct parent/child
  containment edges over derived structure reports under
  `state/structure/<source_id>.yaml`. Rows carry metadata only
  (`source_id`, `parent_symbol_path`, `child_symbol_path`,
  `child_kind`, `child_name`, workspace-relative `report_path`);
  malformed reports, missing or retracted sources, and
  `source_class` / `content_hash` mismatches contribute zero rows
  at build time. `structure_graph_neighbors(workspace, *,
  source_id, symbol_path, direction, limit)` uses the
  `structure_edges` table only to narrow candidate
  `(parent_symbol_path, child_symbol_path)` pairs and rehydrates
  every `StructureGraphEdge` from the **current** report,
  revalidated against the current source registry record; output
  follows the report's deterministic item order. The sidecar is
  derived state only and may be deleted without data loss.
  Inactive claim statuses (retracted, retracted_by_source,
  archived, superseded, stale) and any non-active relation status
  are excluded at build time and filtered again at
  neighbor-lookup time. Commentary, spine prose, rendered page
  prose, raw source bodies, scalar YAML values, comments,
  docstrings, code bodies, journals, merge proposals, and the
  search sidecar are never consulted.

### `cli`
Argparse entry point matching the frozen CLI shape from
`04_specification/operations_and_cli.md`. Installed as the `llloom`
console script via `pyproject.toml`.

### `structured`
Deterministic structured-source extractor owning the
`structure_extract` ingest path. Module `llloom.structured.extract`
defines `StructureExtractError`, `StructureItem`, `StructureReport`,
`extract_structure(...)`, and `write_structure_report(...)`. The
YAML path uses PyYAML's `compose` API (base install). The `code`
path dispatches by `raw_path` suffix to a per-language tree-sitter
walker:

- `.py` → `_load_python_parser` (`tree_sitter` +
  `tree_sitter_python`)
- `.go` → `_load_go_parser` (`tree_sitter` + `tree_sitter_go`)
- `.rs` → `_load_rust_parser` (`tree_sitter` + `tree_sitter_rust`)
- `.ts` → `_load_typescript_parser` (`tree_sitter` +
  `tree_sitter_typescript`'s `language_typescript`)
- `.cs` → `_load_csharp_parser` (`tree_sitter` +
  `tree_sitter_c_sharp`) — added by the C# structured slice
- `.java` → `_load_java_parser` (`tree_sitter` +
  `tree_sitter_java`) — added by Slice 082

Every language loader lazy-imports its grammar inside the loader
function, so importing `llloom` or `llloom.structured` never
requires any structured extra. A missing language wheel raises
`StructureExtractError` whose message names `llloom[structured]`.
Unsupported `code` suffixes (`.tsx`, `.js`, `.kt`, C, C++, etc.)
refuse with a clear message naming the supported suffix set.

A single generic walker (`_walk_node`) drives all six languages;
language-specific node-kind adapters (`_python_node_kind`,
`_go_node_kind`, `_rust_node_kind`, `_typescript_node_kind`,
`_csharp_node_kind`, `_java_node_kind`) identify the
symbol-bearing nodes and emit `(kind, name, nested_prefix)`. The
walker passes ancestor-node-types so Rust's `function_item` inside
an `impl_item` subtree is re-tagged as `method` while preserving
the impl target as the symbol-path prefix; the same pattern lets
C# and Java method declarations qualify under their enclosing
class.

Reports go to `state/structure/<source_id>.yaml` via the standard
temp-file-and-rename pattern and use `code_v1` locators for every
item. The extractor never constructs `SourceDocument`, never
invokes `LLMInvoke`, and never persists scalar values, comments,
docstrings, full source lines, or code bodies. C# structure
extraction also includes a narrow **Unity bridge v1**
classification: a `class_declaration` whose direct base list
contains `MonoBehaviour` (or a qualified name ending in
`.MonoBehaviour`) is re-tagged
`kind == "unity_component"`. The detection is shallow and
textual; no transitive inheritance, no alias resolution. Every
`StructureItem` additionally carries a generic
`tags: tuple[str, ...]` metadata channel (default `()`) that
deterministic classifiers can populate; the Unity bridge
attaches `("framework:unity", "role:component")` to direct
MonoBehaviour subclasses while leaving every other extractor
path empty-tagged. Tags follow a lowercase ASCII
`prefix:value` shape, are surfaced in the serialized
`structure_report_v1` form on every item, and are
rehydration-compatible with reports written before this slice
(`StructureItem.from_mapping` normalises a missing `tags` key
to `()`). The tags channel never replaces `kind`, never drives
verifier or claim semantics, and is not user-configurable in
this slice — it is the reusable hook future framework
classifiers will extend without binding `llloom` core to any
specific framework. Broader language grammars beyond Python, Go,
Rust, TypeScript, and C#; deeper Unity engine semantics
(`ScriptableObject`, lifecycle interpretation, `.unity` /
`.prefab` / `.asmdef` parsing, serialized-field semantics,
editor / IDE integration); claim generation from code beyond
declaration-level and attached-explanation surfaces; user-defined
framework profile files; and MCP structured-ingest tools all
remain deferred.

### `mcp_server`
Optional local stdio MCP server (first slice: read-only and
diagnostic tools only). Installed via the `llloom[mcp]` optional
extra and exposed as the separate `llloom-mcp` console script.
`llloom.mcp_server.tools` owns the pure, SDK-free tool handler
functions (`tool_status`, `tool_query`, `tool_verify`,
`tool_lint`, `tool_graph_neighbors`,
`tool_list_merge_proposals`) plus the `to_jsonable` walker that
converts dataclass results to JSON-compatible dicts;
`llloom.mcp_server.server` wires those handlers into the MCP SDK
with a lazy `from mcp.server.fastmcp import FastMCP` import and
binds one workspace root at process startup. The module is not a
new state owner — every handler wraps an existing library op and
preserves the existing result dataclass shape. No mutating tools,
no model / network path; defence-in-depth check refuses to start
if a forbidden tool name is registered.

## What stays out of the runtime package

- `tests/fixtures/` lives under tests, not in the runtime package.
- No hidden databases, sync daemons, MCP servers, or watch loops.
- No background mutation of canonical state.

## Boundary enforcement points

- `LLMInvoke.invoke` â€” the one place where LLM-allowed inputs are
  enforced. Refuses any input class outside `ALLOWED_READ_CLASSES[op]`
  (e.g. `SourceDocument` on render/query/lint, `SourceSpan` on ingest,
  `ClaimBlockRegion` on ingest). Refuses any write kind outside
  `ALLOWED_WRITE_KINDS[op]`.
- `ops/ingest.py` policy cutoff â€” `index_only` and `structure_extract`
  return BEFORE any `SourceDocument` is constructed or
  `LLMInvoke.invoke` is called. Spy-harness contract test enforces.
- `WorkspaceLock.acquire` â€” refuses when another live op holds the
  lock; stale locks require `reconcile` to consult the journal.
- `WorkspaceLock.is_stale_recoverable` â€” the only predicate `reconcile`
  uses to decide whether to clear a timed-out lock. Requires both
  timeout AND an in-progress journal entry; completed and missing
  journals refuse recovery.
- `ClaimStore.save_entity` â€” atomic write, validates `entity_id`
  pattern.
- `OperationContext` (`ops/_context.py`) â€” every mutating op runs
  inside the context manager so the journal entry exists before the
  lock is taken, the entry is updated with the invocation log on the
  way out, and exceptions leave the entry `in_progress` for reconcile
  to triage.
- `lint` â€” canary enforcement (claim blocks, claim YAMLs, merge
  proposals, journals) and source-evidence hash integrity.

