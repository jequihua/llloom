# Package Scope

`llloom` is the Python package that implements the converged
`04_specification/` design. The workspace remains the knowledge
substrate; the package is the engine that operates on it.

## First vertical slice (Phase 2 current)

The package currently implements the authority core:

- workspace loading and schema validation
- markdown source registration with SHA-256 hashing
- entity-level YAML claim containers (`claims/entities/<id>.yaml`)
- typed source-span locators (`markdown_prose_v1`, `legal_act_v1`) with
  mandatory `excerpt_hash` on every evidence entry
- deterministic span verifier
- variant-(B) page rendering with HTML-comment region markers
- typed-input `LLMInvoke` harness with no filesystem access; per-
  operation read-class and write-kind allow lists enforced
- workspace-scoped file lock with **journal-backed** stale recovery
  (timeout alone never clears a lock)
- persisted `InvocationLog` summaries on every mutating op's journal
  entry (typed-input class + content hash; never raw source text)
- lifecycle and maintenance verbs: `ingest`, `verify`, `render`, `lint`,
  `reconcile`, `unlock`, `promote`, `retract`, `rebuild`,
  `list_merge_proposals`, `review-alias`, `merge-alias`, `reject-alias`,
  and read-only `query`
- canary-enforced exclusion contract
- deterministic verbatim retrieval from registered `index_only`
  sources at the `query` surface; raw `index_only` source bodies
  never enter the `LLMInvoke` harness
- model-backed `claim_extract` ingestion through the typed-input
  harness: structured YAML output is parsed into candidate claims,
  verified against the raw source span, and persisted as a single
  atomic batch; malformed output, unresolvable locators, or hash
  mismatches refuse the whole batch with no partial persistence
- metadata-only **structure context** on narrative `claim_extract`
  via `ingest(..., structure_source_ids=[...])` (CLI:
  `--structure-source <source_id>`). Loads existing structure
  reports under `state/structure/<source_id>.yaml` as
  `StructureItemContext` blocks (source id, source class, language,
  kind, name, symbol path, report path — metadata only) on the
  `LLMInvoke` prompt. Missing or stale requested reports refuse
  cleanly.
- **direct `code_v1` claims on code-backed `claim_extract` and
  `claim_extract_and_view_render`**. The strict YAML parser is
  source-class-aware: narrative source classes admit only their
  narrative locator type, code-backed ingests admit `code_v1`,
  and every emitted `code_v1` candidate must additionally match
  a deterministic admitted span. Two surfaces are admitted:
  declaration-level spans (class / function / method / type /
  interface / trait / enum / struct definitions) and
  **attached explanation spans** (a contiguous line-comment
  block immediately above a declaration, or a Python
  triple-quoted docstring on the line immediately below a class
  / function / async-function declaration). Under
  `claim_extract_and_view_render`, the render step runs only
  after a successful batch persistence and uses the existing
  variant-(B) page contract (renders only inside the
  claim-block region, commentary survives byte-for-byte, no raw
  source body enters render input). Detached comments and
  arbitrary code-body spans remain deferred

## Implemented optional sidecars (post-first-slice)

- **Hybrid search sidecar** (SQLite FTS5) at
  `state/search/search.sqlite`, rebuildable via `llloom rebuild
  search`. Indexes active claim assertions, registered
  non-retracted `index_only` source bodies, and (from the
  structure-report search slice) structure-report item metadata
  read from `state/structure/<source_id>.yaml`. Does not index
  commentary, spine prose, rendered page prose, journals, health
  reports, merge-proposal prose, raw source bodies on the
  structure-report path, or model output. Structure-item rows
  carry metadata only (source id, source class, language, item
  kind, name, symbol path, report path); scalar YAML values,
  comments, docstrings, full source lines, and code bodies are
  never indexed. Never canonical; deletable without data loss.
  `query` uses the sidecar to select candidates when present and
  rehydrates every citation, span, or `StructureItemHit` from
  canonical records (or the on-disk structure report) before
  emitting them.
- **Graph sidecar** (plain SQLite) at
  `state/graph/graph.sqlite`, rebuildable via `llloom rebuild
  graph`. Indexes active relation records from canonical entity
  YAML containers whose endpoint claims both exist and are active;
  does not index commentary, spine prose, rendered page prose, raw
  source bodies, journals, merge proposals, or the search sidecar.
  Never canonical; deletable without data loss. `graph_neighbors(...)`
  uses the sidecar only to narrow candidate relation ids and
  rehydrates every `GraphEdge` from canonical YAML before emission.

## Implemented optional provider adapters (post-first-slice)

- **OpenAI GPT backend** at
  `llloom.llm.openai_backend.OpenAIModelBackend`, installed via
  the `llloom[openai]` optional extra. Implements the existing
  `ModelBackend` protocol; routes through `LLMInvoke`; receives
  only the deterministic prompt the harness assembles; emits
  strict YAML per the existing parser contract. The base install
  does not require the OpenAI SDK. CLI: `llloom ingest <path>
  --model-provider openai --model <model-id>`.

## Implemented optional extractors (post-first-slice)

- **Structured-source ingest** at `llloom.structured`. YAML
  structure extraction runs in the base install via PyYAML.
  Code structure extraction across `.py`, `.go`, `.rs`, `.ts`,
  and `.cs` is gated behind the `llloom[structured]` optional
  extra (tree-sitter plus the five language grammars
  `tree-sitter-python`, `tree-sitter-go`, `tree-sitter-rust`,
  `tree-sitter-typescript`, `tree-sitter-c-sharp`) and each
  grammar is lazy-imported per call so the base install stays
  lean and unsupported `code` suffixes (e.g. `.tsx`, `.js`,
  Java, C, C++) refuse cleanly with a `StructureExtractError`
  naming the supported suffix set and the install extra. C#
  structure extraction also ships a **Unity bridge v1**
  classification: a `class_declaration` directly inheriting
  from `MonoBehaviour` (or a qualified name ending in
  `.MonoBehaviour`) surfaces with
  `kind == "unity_component"`. The rule is direct-base and
  textual only; deeper Unity engine semantics remain deferred.
  Every emitted `StructureItem` additionally carries a generic
  `tags: tuple[str, ...]` metadata channel (defaulting to `()`)
  for framework / role classification; the Unity bridge attaches
  `("framework:unity", "role:component")` to direct
  MonoBehaviour subclasses while every other extractor path
  produces empty-tagged items. Tags follow a lowercase ASCII
  `prefix:value` shape and are surfaced in the serialized
  `structure_report_v1` form on every item;
  `StructureItem.from_mapping` normalises a missing `tags` key
  to `()` so reports written before this slice rehydrate
  cleanly. The tags channel never replaces `kind`, never drives
  verifier or claim semantics, and is not user-configurable in
  this slice. The shared parser-loading boundary
  `_bind_tree_sitter_language(parser, grammar)` accepts both
  already-`tree_sitter.Language` objects and PyCapsule-style
  grammar returns from current `tree_sitter_<lang>.language()`
  packages, wrapping the capsule with `tree_sitter.Language(...)`
  before binding and falling back to the legacy
  `parser.set_language(...)` API for older bindings. Every
  per-language loader routes through this one helper so the
  compatibility shim covers Python, Go, Rust, TypeScript, and
  C# uniformly; missing or unsupported tree-sitter / grammar
  combinations refuse with a single `StructureExtractError`
  naming `llloom[structured]`. `structure_extract` writes a compact,
  deterministic derived report to
  `state/structure/<source_id>.yaml` — non-canonical, deletable
  without data loss. Reports contain structure only; scalar
  values, comments, docstrings, and code bodies are never
  stored. The ingest path never invokes `LLMInvoke`. Supported
  source classes: `structured_yaml`, `code`.

## Implemented optional integrations (post-first-slice)

- **MCP server (first slice)** at `llloom.mcp_server`, installed
  via the `llloom[mcp]` optional extra. Local stdio transport
  only; one workspace bound at startup. Exposes the read-only /
  diagnostic tools `llloom_status`, `llloom_query`,
  `llloom_verify`, `llloom_lint`, `llloom_graph_neighbors`, and
  `llloom_list_merge_proposals`; no mutating tools, no model
  invocation, no network path. Results are JSON-serializable
  copies of existing result dataclasses. Entry point:
  `llloom-mcp --root <workspace>`.

- **PDF working-text prep** at `llloom.pdf_prep`, installed via
  the `llloom[docling]` optional extra. Adds the `prepare-pdf`
  CLI verb and the `llloom.ops.prepare_pdf` library entry,
  producing a deterministic bundle under
  `raw/derived/pdf/<prep_id>/` with three artifacts:
  `docling.md` (the selected ingest artifact), `docling.json`
  (structured-export sibling), and `pdf_prep_manifest.yaml`
  (`pdf_prep_manifest_v1`). The manifest is provider-neutral
  and pins SHA-256 hashes for the source PDF and every produced
  artifact. Future-pipeline component slots (`pymupdf`,
  `grobid`, `pdfplumber`, `nougat`) appear with `status:
  not_run` so a future companion producer can extend the bundle
  without schema migration. The Docling adapter
  (`llloom.pdf_prep.convert_with_docling`) lazy-imports
  `docling` inside the function so a base install without the
  extra still imports `llloom`, builds the CLI parser, and
  passes the default test suite. The op acquires the workspace
  lock and journals through the same `_context.operation(...)`
  pattern as every other mutating op; it never registers
  sources, creates claims, renders pages, or invokes a model.
  Re-running against an existing bundle refuses unless
  `--overwrite` is passed.

## What is NOT in first-party PDF prep

- PyMuPDF integration (page renders, page-coordinate locators,
  forensic preservation) — reserved in the manifest as
  `pymupdf.status: not_run`; intended for a future companion
  producer or a later slice.
- GROBID scholarly-metadata enrichment (TEI XML, authors, refs,
  affiliations) — reserved as `grobid.status: not_run`.
- pdfplumber and Nougat OCR escalation paths — reserved as
  `pdfplumber.status: not_run` and `nougat.status: not_run`.
- PDF-native claim locators or page-coordinate claim
  verification. The selected ingest artifact is ordinary
  Markdown; existing locator and excerpt-hash semantics are
  unchanged.
- Automatic ingest, automatic render, and automatic claim
  extraction. `prepare-pdf` is prep only; the user runs a
  separate `llloom ingest` to register the selected artifact.
- Model invocation from the prep path. Docling is a local
  parser; no provider key is consulted.

## Out of scope in this slice

- vector / semantic search (deferred, separate from FTS sidecar)
- graph visualization, multi-hop query expansion
- broad-language tree-sitter support beyond Python, Go, Rust,
  TypeScript, and C# (Java, C, C++, `.tsx`, `.js`, etc.);
  Unity-specific engine semantics beyond the Unity bridge v1
  classification (direct `MonoBehaviour` subclasses surface
  as `unity_component`): `ScriptableObject` classification,
  lifecycle interpretation, `.unity` / `.prefab` / `.asmdef`
  parsing, serialized-field semantics, asset reference
  reasoning, editor / IDE integration; detached or
  free-floating
  comment / docstring claims; arbitrary code-body-span claims;
  MCP structured ingest tools
  (search-sidecar indexing of structure-report metadata,
  `query` surfacing of rehydrated `StructureItemHit` records,
  graph-sidecar indexing of direct parent/child containment
  via `structure_graph_neighbors(...)`, metadata-only structure
  context on narrative `claim_extract` via
  `--structure-source` / `structure_source_ids=[...]`, and
  direct `code_v1` claims on code-backed `claim_extract` and
  `claim_extract_and_view_render` for declaration-level spans
  plus attached explanation spans are implemented)
- Gemini, local-model, Ollama, LiteLLM adapters (Anthropic is
  implemented behind the `llloom[anthropic]` optional extra)
- multi-provider orchestration, provider routing, retry queues,
  streaming, tool calling, hosted retrieval, background jobs
- mutating MCP tools, HTTP / SSE / WebSocket MCP transport,
  hosted MCP service, background watch mode, editor plugins
- query filing back into the repo
- autonomous background ingestion
- broad file-format ingest beyond markdown
- multi-repo registry
- editor integrations

These are deferred per `04_specification/package_boundary.md`.

## Reviewability bar

Every public operation returns a dataclass result. The CLI, the library
API, and later wrappers consume the same shapes. Refusal conditions are
visible in the result object (`refusal_reason`, `refused`, `failures`,
`canary_hits`). A future architect-reviewer can trace any behavior back
to a spec file listed in `04_specification/`.

