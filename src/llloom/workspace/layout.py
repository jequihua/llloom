"""Workspace layout resolution and validation.

Implements the canonical layout from
``04_specification/storage_and_state_model.md`` §"Canonical workspace layout".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REQUIRED_DIRS = (
    "raw/sources",
    "raw/assets",
    "claims/entities",
    "claims/merge_proposals",
    "pages/entities",
    "pages/concepts",
    "pages/syntheses",
    "pages/navigation",
    "schema",
    "schema/prompts",
    "state/source_registry",
    "state/locks",
    "state/journals",
    "state/transactions",
    "state/reports/health",
    "state/reports/updates",
    "state/rebuild",
)

REQUIRED_SCHEMA_FILES = (
    "schema/source_classes.yaml",
    "schema/ingest_policies.yaml",
    "schema/page_classes.yaml",
    "schema/spine_manifest.yaml",
)


class WorkspaceError(Exception):
    """Raised when workspace layout or schema is malformed."""


@dataclass(frozen=True)
class Workspace:
    """Resolved workspace paths.

    The package does not hide on-disk layout from callers. Every canonical
    directory has a named attribute so callers can be explicit about which
    part of the workspace they touch.
    """

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def raw_sources(self) -> Path:
        return self.root / "raw" / "sources"

    @property
    def claims(self) -> Path:
        return self.root / "claims"

    @property
    def claims_entities(self) -> Path:
        return self.root / "claims" / "entities"

    @property
    def claims_merge_proposals(self) -> Path:
        return self.root / "claims" / "merge_proposals"

    @property
    def pages(self) -> Path:
        return self.root / "pages"

    @property
    def schema(self) -> Path:
        return self.root / "schema"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def state_source_registry(self) -> Path:
        return self.root / "state" / "source_registry"

    @property
    def state_locks(self) -> Path:
        return self.root / "state" / "locks"

    @property
    def state_journals(self) -> Path:
        return self.root / "state" / "journals"

    @property
    def state_transactions(self) -> Path:
        """In-flight write buffers for transactional operations.

        Slice 074 added ``state/transactions/<op_id>/`` as the staging
        root for render commit. Directories under this path are
        **in-flight write buffers**, not rebuildable sidecars — they
        carry authoritative work-in-progress between staging and
        commit, are removed by the success path, and are left on disk
        with a diagnosable manifest when the operation is interrupted
        so ``reconcile`` (or an operator) can triage. See
        ``04_specification/storage_and_state_model.md`` for the full
        contract.
        """
        return self.root / "state" / "transactions"

    @property
    def state_reports_health(self) -> Path:
        return self.root / "state" / "reports" / "health"

    @property
    def state_reports_updates(self) -> Path:
        """Durable audit evidence for mutating seed-apply operations.

        Slice 076 added ``state/reports/updates/<op_id>.yaml`` as the
        canonical machine-readable record for each real mutating seed
        apply. Reports include manifest + source hashes, planned /
        created claim ids, entities touched, pages rendered, before /
        after counts, bounded excerpt previews, and provenance fields
        proving the path never invoked a model. Reports are durable
        audit evidence, not rebuildable sidecars — they encode work
        that already happened. Dry-run writes nothing here. See
        ``04_specification/storage_and_state_model.md`` for the full
        contract.
        """
        return self.root / "state" / "reports" / "updates"

    @property
    def state_rebuild(self) -> Path:
        return self.root / "state" / "rebuild"

    @property
    def state_search(self) -> Path:
        return self.root / "state" / "search"

    @property
    def search_db(self) -> Path:
        return self.root / "state" / "search" / "search.sqlite"

    @property
    def state_graph(self) -> Path:
        return self.root / "state" / "graph"

    @property
    def graph_db(self) -> Path:
        return self.root / "state" / "graph" / "graph.sqlite"

    @property
    def state_structure(self) -> Path:
        return self.root / "state" / "structure"

    def structure_report_path(self, source_id: str) -> Path:
        return self.state_structure / f"{source_id}.yaml"

    @property
    def render_fingerprints(self) -> Path:
        return self.root / "state" / "render_fingerprints.yaml"

    def validate(self) -> None:
        """Validate that every required directory and schema file exists.

        Raises WorkspaceError listing every missing entry.
        """
        missing: list[str] = []
        for rel in REQUIRED_DIRS:
            if not (self.root / rel).is_dir():
                missing.append(f"missing directory: {rel}")
        for rel in REQUIRED_SCHEMA_FILES:
            if not (self.root / rel).is_file():
                missing.append(f"missing schema file: {rel}")
        if missing:
            raise WorkspaceError(
                "workspace layout invalid: " + "; ".join(missing)
            )

    @classmethod
    def load(cls, root: Path | str) -> "Workspace":
        """Load and validate a workspace at ``root``."""
        ws = cls(root=Path(root).resolve())
        ws.validate()
        return ws

    @classmethod
    def init(cls, root: Path | str) -> "Workspace":
        """Create a fresh workspace at ``root`` with starter schema.

        Implements the ``init`` operation contract from
        ``04_specification/operations_and_cli.md``.
        """
        root_path = Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        for rel in REQUIRED_DIRS:
            (root_path / rel).mkdir(parents=True, exist_ok=True)

        _seed_schema_files(root_path)
        _seed_spine_files(root_path)
        _seed_render_fingerprints(root_path)

        return cls.load(root_path)


def _seed_schema_files(root: Path) -> None:
    """Write starter schema files if absent."""
    files: dict[str, str] = {
        "schema/source_classes.yaml": _STARTER_SOURCE_CLASSES,
        "schema/ingest_policies.yaml": _STARTER_INGEST_POLICIES,
        "schema/page_classes.yaml": _STARTER_PAGE_CLASSES,
        "schema/spine_manifest.yaml": _STARTER_SPINE_MANIFEST,
    }
    for rel, content in files.items():
        path = root / rel
        if not path.exists():
            path.write_text(content, encoding="utf-8")


def _seed_spine_files(root: Path) -> None:
    """Seed the minimal human-owned editorial spine."""
    overview = root / "pages" / "overview.md"
    if not overview.exists():
        overview.write_text(_STARTER_OVERVIEW, encoding="utf-8")


def _seed_render_fingerprints(root: Path) -> None:
    fp = root / "state" / "render_fingerprints.yaml"
    if not fp.exists():
        fp.write_text("fingerprints: {}\n", encoding="utf-8")


_STARTER_SOURCE_CLASSES = """\
# Registered source classes. Each source class chooses a locator shape.
classes:
  markdown_prose:
    locator: markdown_prose_v1
    description: Prose markdown (scientific articles, narrative documents).
  legal_act:
    locator: legal_act_v1
    description: Legal or policy markdown with sections and clauses.
  code:
    locator: code_v1
    description: Python/code source for deterministic structure extraction (requires llloom[structured]).
  structured_yaml:
    locator: code_v1
    description: Structured YAML source for deterministic structure extraction.
  raw_evidence:
    locator: markdown_prose_v1
    description: |
      Neutral starter source class (Slice 083) for unsupported or
      intentionally unstructured UTF-8 evidence registered for hash
      + exact deterministic retrieval only. Maps to the existing
      `index_only` ingest policy: no claims, no pages, no structure
      reports, no model invocation. Reuses the `markdown_prose_v1`
      locator shape for schema compatibility only — a neutral
      locator type is not introduced in this slice. Prefer `code`
      when structured extraction is available; use `raw_evidence`
      for `.java` (pre-Java-structure-slice), `.kt`, `.proto`, or
      any UTF-8 text whose structure llloom does not yet parse.
"""

_STARTER_INGEST_POLICIES = """\
# Ingest policy assignment. Maps source class to policy.
# Policies: deny | index_only | structure_extract | claim_extract | claim_extract_and_view_render
policies:
  markdown_prose: claim_extract_and_view_render
  legal_act: claim_extract
  code: structure_extract
  structured_yaml: structure_extract
  raw_evidence: index_only
defaults:
  unknown: deny
"""

_STARTER_PAGE_CLASSES = """\
# Page classes. The first slice keeps this minimal.
classes:
  entity:
    directory: pages/entities
  concept:
    directory: pages/concepts
  synthesis:
    directory: pages/syntheses
  navigation:
    directory: pages/navigation
"""

_STARTER_SPINE_MANIFEST = """\
# Editorial spine manifest. Files listed here are human-owned and
# refused as direct-write targets by LLMInvoke.
spine_files:
  - pages/overview.md
  - schema/source_classes.yaml
  - schema/ingest_policies.yaml
  - schema/page_classes.yaml
  - schema/spine_manifest.yaml
spine_globs:
  - pages/navigation/**
"""

_STARTER_OVERVIEW = """\
---
page_id: overview
page_class: navigation
write_policy: human
status: human_authored
---

# Overview

Human-authored entry point for the knowledge compiler.

This page is part of the editorial spine. The LLM-invocation harness
refuses to write to this file during authoritative operations.
"""
