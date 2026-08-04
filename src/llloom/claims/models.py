"""Typed models for claim containers and their dependents.

Mirror the worked example in
``04_specification/storage_and_state_model.md`` §"Canonical claim container choice".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Lifecycle states frozen for the first slice.
CLAIM_STATUSES = frozenset(
    {
        "draft",
        "reviewed",
        "validated",
        "superseded",
        "archived",
        "retracted",
        "stale",
        "retracted_by_source",
    }
)

RELATION_TYPES = frozenset(
    {"supports", "contradicts", "supersedes", "refines", "about"}
)

VERIFICATION_STATUSES = frozenset({"unverified", "verified", "failed"})

ENTITY_STATUSES = frozenset({"active", "merged_into", "retired"})


@dataclass
class Locator:
    """Source-span locator. Shape depends on ``locator_type``.

    The first slice supports ``markdown_prose_v1`` and ``legal_act_v1``
    fully; ``code_v1`` is reserved for the deferred structured-source path.
    """

    locator_type: str
    # markdown_prose_v1 / legal_act_v1 share these:
    heading_path: list[str] | None = None
    paragraph_index: int | None = None
    sentence_start: int | None = None
    sentence_end: int | None = None
    # legal_act_v1 only:
    act_title: str | None = None
    section_label: str | None = None
    clause_label: str | None = None
    # code_v1 (reserved):
    path: str | None = None
    symbol_path: str | None = None
    start_line: int | None = None
    start_col: int | None = None
    end_line: int | None = None
    end_col: int | None = None

    def to_mapping(self) -> dict[str, Any]:
        data: dict[str, Any] = {"locator_type": self.locator_type}
        for name in (
            "heading_path",
            "paragraph_index",
            "sentence_start",
            "sentence_end",
            "act_title",
            "section_label",
            "clause_label",
            "paragraph_index",
            "path",
            "symbol_path",
            "start_line",
            "start_col",
            "end_line",
            "end_col",
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Locator":
        kwargs = {k: v for k, v in data.items() if k != "locator_type"}
        return cls(locator_type=data["locator_type"], **kwargs)


@dataclass
class Evidence:
    source_id: str
    locator: Locator
    excerpt_hash: str
    excerpt: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_id": self.source_id,
            "locator": self.locator.to_mapping(),
            "excerpt_hash": self.excerpt_hash,
        }
        if self.excerpt is not None:
            out["excerpt"] = self.excerpt
        return out

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Evidence":
        return cls(
            source_id=str(data["source_id"]),
            locator=Locator.from_mapping(dict(data["locator"])),
            excerpt_hash=str(data["excerpt_hash"]),
            excerpt=data.get("excerpt"),
        )


@dataclass
class RenderTarget:
    page_id: str
    block_id: str

    def to_mapping(self) -> dict[str, Any]:
        return {"page_id": self.page_id, "block_id": self.block_id}

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RenderTarget":
        return cls(page_id=str(data["page_id"]), block_id=str(data["block_id"]))


@dataclass
class Assertion:
    claim_id: str
    subject_id: str
    claim_kind: str
    claim_text: str
    evidence: list[Evidence] = field(default_factory=list)
    render_targets: list[RenderTarget] = field(default_factory=list)
    status: str = "draft"
    verification_status: str = "unverified"
    supersedes: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "claim_id": self.claim_id,
            "subject_id": self.subject_id,
            "claim_kind": self.claim_kind,
            "claim_text": self.claim_text,
            "status": self.status,
            "verification_status": self.verification_status,
            "evidence": [e.to_mapping() for e in self.evidence],
            "render_targets": [t.to_mapping() for t in self.render_targets],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.supersedes is not None:
            out["supersedes"] = self.supersedes
        return out

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Assertion":
        return cls(
            claim_id=str(data["claim_id"]),
            subject_id=str(data["subject_id"]),
            claim_kind=str(data["claim_kind"]),
            claim_text=str(data["claim_text"]),
            evidence=[Evidence.from_mapping(dict(e)) for e in data.get("evidence", [])],
            render_targets=[
                RenderTarget.from_mapping(dict(t))
                for t in data.get("render_targets", [])
            ],
            status=str(data.get("status", "draft")),
            verification_status=str(data.get("verification_status", "unverified")),
            supersedes=data.get("supersedes"),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )


@dataclass
class Relation:
    relation_id: str
    source_claim_id: str
    relation_type: str
    target_claim_id: str
    status: str = "active"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_claim_id": self.source_claim_id,
            "relation_type": self.relation_type,
            "target_claim_id": self.target_claim_id,
            "status": self.status,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Relation":
        return cls(
            relation_id=str(data["relation_id"]),
            source_claim_id=str(data["source_claim_id"]),
            relation_type=str(data["relation_type"]),
            target_claim_id=str(data["target_claim_id"]),
            status=str(data.get("status", "active")),
        )


@dataclass
class Alias:
    alias_id: str
    alias_text: str
    status: str = "active"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "alias_id": self.alias_id,
            "alias_text": self.alias_text,
            "status": self.status,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Alias":
        return cls(
            alias_id=str(data["alias_id"]),
            alias_text=str(data["alias_text"]),
            status=str(data.get("status", "active")),
        )


@dataclass
class EntityContainer:
    """One YAML file per entity under ``claims/entities/``."""

    entity_id: str
    entity_type: str
    display_name: str
    status: str = "active"
    assertions: list[Assertion] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    aliases: list[Alias] = field(default_factory=list)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "display_name": self.display_name,
            "status": self.status,
            "aliases": [a.to_mapping() for a in self.aliases],
            "assertions": [a.to_mapping() for a in self.assertions],
            "relations": [r.to_mapping() for r in self.relations],
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EntityContainer":
        return cls(
            entity_id=str(data["entity_id"]),
            entity_type=str(data["entity_type"]),
            display_name=str(data["display_name"]),
            status=str(data.get("status", "active")),
            assertions=[
                Assertion.from_mapping(dict(a)) for a in data.get("assertions", [])
            ],
            relations=[Relation.from_mapping(dict(r)) for r in data.get("relations", [])],
            aliases=[Alias.from_mapping(dict(a)) for a in data.get("aliases", [])],
        )

    def find_assertion(self, claim_id: str) -> Assertion | None:
        for a in self.assertions:
            if a.claim_id == claim_id:
                return a
        return None


@dataclass
class MergeProposal:
    """Write-as-new, queue-for-merge alias review artifact."""

    proposal_id: str
    source_entity_id: str  # the newly-written entity under review
    target_entity_id: str  # the candidate existing entity to merge into
    proposed_alias_text: str
    status: str = "pending"  # pending | approved | rejected | applied
    reason: str = ""
    created_at: str = ""
    reviewed_at: str | None = None
    review_notes: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "proposal_id": self.proposal_id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "proposed_alias_text": self.proposed_alias_text,
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at,
        }
        if self.reviewed_at is not None:
            out["reviewed_at"] = self.reviewed_at
        if self.review_notes is not None:
            out["review_notes"] = self.review_notes
        return out

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "MergeProposal":
        return cls(
            proposal_id=str(data["proposal_id"]),
            source_entity_id=str(data["source_entity_id"]),
            target_entity_id=str(data["target_entity_id"]),
            proposed_alias_text=str(data["proposed_alias_text"]),
            status=str(data.get("status", "pending")),
            reason=str(data.get("reason", "")),
            created_at=str(data.get("created_at", "")),
            reviewed_at=data.get("reviewed_at"),
            review_notes=data.get("review_notes"),
        )
