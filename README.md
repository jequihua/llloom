# llloom

`llloom` is a source-grounded memory system for language-model agents. It turns
durable project evidence — architecture decisions, code facts, source excerpts,
standards, notes, reviewed findings, open questions — into verified claims,
derived indexes, and wiki-style pages, while keeping an audit trail for every
mutating operation.

Use it when an agent needs memory that is better than chat history: every claim
is tied to the source bytes it came from, and every write is journaled.

## Install

Requires Python 3.11+. The base runtime dependency is PyYAML. Install the
package from this repository:

```bash
python -m pip install .
```

Optional product extras:

```bash
python -m pip install ".[structured]"   # tree-sitter code structure extraction
python -m pip install ".[docling]"      # PDF working-text prep
python -m pip install ".[openai]"       # OpenAI provider adapter
python -m pip install ".[anthropic]"    # Anthropic provider adapter
python -m pip install ".[mcp]"          # read-only MCP server
```

The base install is offline and needs no provider account. Contributors who
want to run the test suite should instead use an editable install with the
development extra (see `testing_strategy.md`):

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

Create and exercise a disposable workspace. The sequence first writes a small
source file with the Python standard library, then registers it:

```bash
llloom --root ./memory init
llloom --root ./memory status
python -c "from pathlib import Path; p = Path('./memory/raw/sources'); p.mkdir(parents=True, exist_ok=True); (p / 'architecture.md').write_text('# Architecture\n\nThe render transaction staging area holds rendered output before commit.\n', encoding='utf-8')"
llloom --root ./memory ingest raw/sources/architecture.md \
  --source-id project.architecture \
  --source-class markdown_prose
llloom --root ./memory page create concept/demo --title "Demo"
llloom --root ./memory render --dry-run
llloom --root ./memory query "render transaction staging"
llloom --root ./memory doctor
```

The default extractor on `markdown_prose` sources registers the source without
creating claims, so this works fully offline. Provider-backed extraction and
deterministic seed manifests are available when you want claims populated.

## Safety Model

- **Sources are immutable evidence.** A source id is bound to its bytes and
  source class; reusing an id with different bytes is refused.
- **Claims are verified.** Claims carry source locators and excerpt hashes;
  verification re-resolves them against the staged raw source.
- **Pages are views.** Rendered claim blocks are generated from canonical claim
  YAML. Human commentary regions are preserved across renders.
- **Mutations are journaled.** Workspace writes take a lock and an operation
  journal; interrupted operations can be diagnosed and reconciled.
- **Sidecars are derived.** Search, graph, structure, and render-fingerprint
  artifacts can be rebuilt from canonical state.
- **Model use is explicit.** Optional providers pass through a typed-input
  harness, strict parsing, verification, and batch-atomic persistence.

## CLI Overview

Commands group by purpose: setup (`init`); read-only inspection (`status`,
`query`, `lint`, `verify`, `doctor`, and listing commands); mutating writes
(`ingest`, `seed`, `render`, `page`, lifecycle commands); maintenance
(`reconcile`, `unlock`, `rebuild`); and optional helpers such as `prepare-pdf`.
Run `llloom --help` for the current verb list and `llloom <verb> --help` for
per-verb options.

## Pages

Create a legacy page deterministically (writes one valid stub, refuses to
overwrite):

```bash
llloom --root ./memory page create concept/legacy-demo --title "Legacy Demo"
```

An explicit opt-in creates one interoperable page carrying a pinned framework
profile block:

```bash
llloom --root ./memory page create concept/profiled-demo --title "Profiled Demo" --framework-profile 0.1-rc.1
```

Omitting `--framework-profile` preserves the legacy creation bytes and
behavior exactly; the opt-in is validated before publication and never changes
default behavior.

## Optional Features

- **Model providers** (`openai`, `anthropic` extras): provider output is parsed
  strictly, verified against source spans, and persisted atomically. The default
  CLI path never requires one.
- **Structured code extraction** (`structured` extra): deterministic,
  metadata-only structure reports for supported source files.
- **PDF working-text prep** (`docling` extra): deterministic text bundles under
  `raw/derived/pdf/`; it never registers sources or invokes a model.
- **MCP server** (`mcp` extra): a read-only/diagnostic MCP surface — status,
  query, verify, lint, graph neighbors, and merge-proposal listing. Mutating
  tools are not exposed.

## Contributing

Run the product test suite (see `testing_strategy.md` for the layout and the
accepted Windows partition):

```bash
python -m pytest
```

Product contracts live in `architecture_contract.md`, `public_api_contract.md`,
`package_scope.md`, and `testing_strategy.md` at the repository root.
