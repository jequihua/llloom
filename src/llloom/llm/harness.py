"""Typed-input LLMInvoke harness.

Enforcement model (from ``04_specification/component_contracts.md``
§LLMInvoke):

- The harness accepts only typed input objects.
- The harness has **no filesystem access** and no network access. It
  cannot resolve a path to file bytes.
- Excluded content classes (commentary regions, spine prose, derived
  files, ``index_only`` source bodies) are unreachable by construction
  because no typed input class represents them.
- Every invocation writes a log record.

Operation matrix lives in
``04_specification/component_contracts.md`` §LLMInvoke.

The first slice ships a :class:`NullModel` backend that produces no
output. Real model bindings can be added later by implementing the
:class:`ModelBackend` protocol. The authority-core contracts in this
slice do not require a live model: claim extraction, rendering, and
query all have deterministic code paths that construct typed inputs and
pass them through the harness for audit/canary purposes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol


# ---- typed input objects ------------------------------------------------


OperationKind = str  # "ingest" | "render" | "query" | "lint"


ALLOWED_OPERATIONS = frozenset({"ingest", "render", "query", "lint"})


@dataclass(frozen=True)
class SourceDocument:
    """Raw source content prepared by a policy-checked caller.

    The harness never opens files; the caller is responsible for reading
    the bytes and constructing this object.
    """

    source_id: str
    source_class: str
    text: str

    @property
    def content_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    entity_id: str
    claim_text: str

    @property
    def content_hash(self) -> str:
        payload = f"{self.claim_id}\n{self.entity_id}\n{self.claim_text}"
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceSpan:
    """Verbatim span returned from deterministic exact retrieval.

    Used for ``query`` answers that cite ``index_only`` sources verbatim.
    """

    source_id: str
    excerpt: str

    @property
    def content_hash(self) -> str:
        payload = f"{self.source_id}\n{self.excerpt}"
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClaimBlockRegion:
    """Pre-extracted authoritative region of a rendered page.

    The *only* page content class the harness knows about. Commentary
    regions cannot be represented here because there is no such class.
    """

    page_id: str
    block_id: str
    rendered_text: str

    @property
    def content_hash(self) -> str:
        payload = f"{self.page_id}\n{self.block_id}\n{self.rendered_text}"
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SchemaDocument:
    name: str
    text: str

    @property
    def content_hash(self) -> str:
        payload = f"{self.name}\n{self.text}"
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WriteTarget:
    """A declared write target inside ``claims/`` or ``state/``.

    Must be a repo-relative POSIX path.
    """

    path: str
    kind: str  # "claim_entity" | "merge_proposal" | "render_fingerprint" | "journal"


@dataclass(frozen=True)
class StructureItemContext:
    """Metadata-only summary of one structure-report item.

    Carries structure metadata only — no raw code text, comments,
    docstrings, scalar values, or ``code_v1`` excerpt bytes. Used as
    explicit, caller-selected context on ``claim_extract`` /
    ``claim_extract_and_view_render`` ingest of a *narrative* source;
    persisted claims must still ground in that narrative source, not
    in the structure source. ``StructureItemContext`` is allowed only
    for ``operation_kind == "ingest"``; render, query, and lint refuse
    it.
    """

    source_id: str
    source_class: str
    language: str
    kind: str
    name: str
    symbol_path: str
    report_path: str

    @property
    def content_hash(self) -> str:
        payload = "\n".join(
            (
                self.source_id,
                self.source_class,
                self.language,
                self.kind,
                self.name,
                self.symbol_path,
                self.report_path,
            )
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Operation matrix.
#
# Mirrors the table in
# ``04_specification/component_contracts.md`` §LLMInvoke.
# Read-class enforcement is structural: any input class not in the
# allow-list for ``operation_kind`` is refused. Excluded content classes
# (commentary regions, spine prose, ``index_only`` source bodies) are
# unreachable by construction because they have no typed input class at
# all.

ALLOWED_READ_CLASSES: dict[OperationKind, frozenset[str]] = {
    # Ingest: source under ingest, existing claim records for affected
    # entities, schema, optional metadata-only structure context.
    # NEVER claim-block regions (M2). NEVER source spans (those are
    # deterministic exact-retrieval results, used by query).
    "ingest": frozenset(
        {
            "SourceDocument",
            "ClaimRecord",
            "SchemaDocument",
            "StructureItemContext",
        }
    ),
    # Render: claims and schema and explicitly-allowed claim-block
    # metadata. NEVER raw source bodies (renderer is claim-only).
    "render": frozenset({"ClaimRecord", "SchemaDocument", "ClaimBlockRegion"}),
    # Query: claims, allowed claim-block regions, schema, deterministic
    # verbatim spans. NEVER raw source bodies.
    "query": frozenset(
        {"ClaimRecord", "SchemaDocument", "ClaimBlockRegion", "SourceSpan"}
    ),
    # Lint LLM-backed advisory checks: claims and schema only.
    "lint": frozenset({"ClaimRecord", "SchemaDocument"}),
}

ALLOWED_WRITE_KINDS: dict[OperationKind, frozenset[str]] = {
    "ingest": frozenset({"claim_entity", "merge_proposal", "journal"}),
    "render": frozenset({"claim_block", "render_fingerprint", "journal"}),
    "query": frozenset(),  # first slice: query is read-only
    "lint": frozenset({"report", "journal"}),
}


# ---- invocation log -----------------------------------------------------


@dataclass
class InvocationLog:
    """Audit record for a single harness invocation."""

    op_id: str
    operation_kind: OperationKind
    model_identifier: str
    read_inputs: list[dict] = field(default_factory=list)
    write_targets: list[dict] = field(default_factory=list)
    output_hash: str | None = None
    started_at: str = ""
    completed_at: str = ""
    refusal: str | None = None

    def to_mapping(self) -> dict:
        out = {
            "op_id": self.op_id,
            "operation_kind": self.operation_kind,
            "model_identifier": self.model_identifier,
            "read_inputs": list(self.read_inputs),
            "write_targets": list(self.write_targets),
            "output_hash": self.output_hash,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.refusal is not None:
            out["refusal"] = self.refusal
        return out


# ---- model backend protocol --------------------------------------------


class ModelBackend(Protocol):
    """Pluggable model backend.

    The harness serializes typed inputs into a deterministic prompt and
    calls ``generate(prompt)``. Backends must not open files or make
    network calls the harness did not authorize.
    """

    identifier: str

    def generate(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


class NullModel:
    """Backend that returns no text.

    Suitable for the first-slice contract tests: it exercises the
    harness's input-boundary and logging guarantees without producing
    content that could leak.
    """

    identifier = "null-model/v0"

    def generate(self, prompt: str) -> str:
        # Deliberately returns the empty string.
        _ = prompt
        return ""


# ---- refusal + harness --------------------------------------------------


class HarnessRefusal(Exception):
    """Raised when the harness refuses an invocation.

    Never subclass to silently fall back; refusals are the contract.
    """


class LLMInvoke:
    """The single authorized gateway for LLM-backed operations.

    The harness has no filesystem access. Callers must construct the
    typed inputs themselves, which requires a policy check before
    loading bytes into a typed object.
    """

    def __init__(self, model: ModelBackend | None = None) -> None:
        self._model = model or NullModel()

    def invoke(
        self,
        *,
        op_id: str,
        operation_kind: OperationKind,
        source_documents: list[SourceDocument] | None = None,
        claim_records: list[ClaimRecord] | None = None,
        source_spans: list[SourceSpan] | None = None,
        claim_blocks: list[ClaimBlockRegion] | None = None,
        schema_documents: list[SchemaDocument] | None = None,
        structure_items: list[StructureItemContext] | None = None,
        write_targets: list[WriteTarget] | None = None,
    ) -> tuple[str, InvocationLog]:
        """Run one authorized LLM invocation.

        Returns (output_text, log).
        """
        if operation_kind not in ALLOWED_OPERATIONS:
            raise HarnessRefusal(
                f"operation_kind {operation_kind!r} not in {sorted(ALLOWED_OPERATIONS)}"
            )

        source_documents = list(source_documents or [])
        claim_records = list(claim_records or [])
        source_spans = list(source_spans or [])
        claim_blocks = list(claim_blocks or [])
        schema_documents = list(schema_documents or [])
        structure_items = list(structure_items or [])
        write_targets = list(write_targets or [])

        self._check_operation_matrix(
            operation_kind=operation_kind,
            source_documents=source_documents,
            claim_records=claim_records,
            source_spans=source_spans,
            claim_blocks=claim_blocks,
            schema_documents=schema_documents,
            structure_items=structure_items,
            write_targets=write_targets,
        )

        prompt = self._assemble_prompt(
            operation_kind,
            source_documents,
            claim_records,
            source_spans,
            claim_blocks,
            schema_documents,
            structure_items,
        )

        started_at = _iso_now()
        output = self._model.generate(prompt)
        completed_at = _iso_now()

        output_hash = (
            "sha256:" + hashlib.sha256(output.encode("utf-8")).hexdigest()
        )

        log = InvocationLog(
            op_id=op_id,
            operation_kind=operation_kind,
            model_identifier=self._model.identifier,
            read_inputs=_read_inputs_summary(
                source_documents,
                claim_records,
                source_spans,
                claim_blocks,
                schema_documents,
                structure_items,
            ),
            write_targets=[
                {"path": t.path, "kind": t.kind} for t in write_targets
            ],
            output_hash=output_hash,
            started_at=started_at,
            completed_at=completed_at,
        )
        return output, log

    # ---- internals ------------------------------------------------------

    @staticmethod
    def _check_operation_matrix(
        *,
        operation_kind: OperationKind,
        source_documents: list[SourceDocument],
        claim_records: list[ClaimRecord],
        source_spans: list[SourceSpan],
        claim_blocks: list[ClaimBlockRegion],
        schema_documents: list[SchemaDocument],
        structure_items: list[StructureItemContext],
        write_targets: list[WriteTarget],
    ) -> None:
        """Enforce the per-operation read- and write-class allow-lists.

        Read-class enforcement comes from
        ``04_specification/component_contracts.md`` §LLMInvoke. Excluded
        content classes (commentary, spine prose, ``index_only`` source
        bodies) are unreachable by construction because they have no
        typed input class; the allow-list defends against the cases that
        DO have a typed class but are inappropriate for the operation
        (e.g. a raw ``SourceDocument`` reaching ``render``).
        """
        allowed_reads = ALLOWED_READ_CLASSES.get(operation_kind, frozenset())
        present: list[tuple[str, int]] = [
            ("SourceDocument", len(source_documents)),
            ("ClaimRecord", len(claim_records)),
            ("SourceSpan", len(source_spans)),
            ("ClaimBlockRegion", len(claim_blocks)),
            ("SchemaDocument", len(schema_documents)),
            ("StructureItemContext", len(structure_items)),
        ]
        for class_name, count in present:
            if count and class_name not in allowed_reads:
                raise HarnessRefusal(
                    f"operation {operation_kind!r} may not receive "
                    f"{class_name} inputs; allowed read classes: "
                    f"{sorted(allowed_reads)}"
                )

        allowed_writes = ALLOWED_WRITE_KINDS.get(operation_kind, frozenset())
        for wt in write_targets:
            if wt.kind not in allowed_writes:
                raise HarnessRefusal(
                    f"operation {operation_kind!r} may not write kind {wt.kind!r}; "
                    f"allowed: {sorted(allowed_writes)}"
                )

    @staticmethod
    def _assemble_prompt(
        operation_kind: OperationKind,
        sources: list[SourceDocument],
        claims: list[ClaimRecord],
        spans: list[SourceSpan],
        blocks: list[ClaimBlockRegion],
        schemas: list[SchemaDocument],
        structure_items: list[StructureItemContext],
    ) -> str:
        """Serialize typed inputs into a deterministic prompt.

        This is NOT a template for production model use. It is a
        deterministic serialization whose only job is to be auditable.
        """
        parts: list[str] = [f"# operation: {operation_kind}"]
        for s in sources:
            parts.append(f"## source {s.source_id} [{s.source_class}] hash={s.content_hash}")
            parts.append(s.text)
        for c in claims:
            parts.append(f"## claim {c.claim_id} entity={c.entity_id} hash={c.content_hash}")
            parts.append(c.claim_text)
        for sp in spans:
            parts.append(f"## span {sp.source_id} hash={sp.content_hash}")
            parts.append(sp.excerpt)
        for b in blocks:
            parts.append(
                f"## claim_block {b.page_id}:{b.block_id} hash={b.content_hash}"
            )
            parts.append(b.rendered_text)
        for sc in schemas:
            parts.append(f"## schema {sc.name} hash={sc.content_hash}")
            parts.append(sc.text)
        for si in structure_items:
            parts.append(
                f"## structure_item {si.source_id} [{si.language}] "
                f"kind={si.kind} symbol={si.symbol_path} hash={si.content_hash}"
            )
            parts.append(f"name: {si.name}\nreport: {si.report_path}")
        return "\n\n".join(parts)


def _read_inputs_summary(
    sources: list[SourceDocument],
    claims: list[ClaimRecord],
    spans: list[SourceSpan],
    blocks: list[ClaimBlockRegion],
    schemas: list[SchemaDocument],
    structure_items: list[StructureItemContext],
) -> list[dict]:
    """Summarize typed inputs for the invocation log."""
    out: list[dict] = []
    for s in sources:
        out.append({"class": "SourceDocument", "id": s.source_id, "hash": s.content_hash})
    for c in claims:
        out.append({"class": "ClaimRecord", "id": c.claim_id, "hash": c.content_hash})
    for sp in spans:
        out.append({"class": "SourceSpan", "id": sp.source_id, "hash": sp.content_hash})
    for b in blocks:
        out.append(
            {
                "class": "ClaimBlockRegion",
                "id": f"{b.page_id}:{b.block_id}",
                "hash": b.content_hash,
            }
        )
    for sc in schemas:
        out.append({"class": "SchemaDocument", "id": sc.name, "hash": sc.content_hash})
    for si in structure_items:
        out.append(
            {
                "class": "StructureItemContext",
                "id": f"{si.source_id}:{si.symbol_path}",
                "hash": si.content_hash,
            }
        )
    return out


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
