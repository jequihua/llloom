"""Entity claim container persistence.

One YAML file per entity under ``claims/entities/``. Merge proposals live
separately under ``claims/merge_proposals/``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import yaml

from llloom.claims.models import EntityContainer, MergeProposal
from llloom.workspace.layout import Workspace


ENTITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{1,127}$")


class ClaimStoreError(Exception):
    """Raised for claim-store-level failures."""


class ClaimStore:
    """File-backed entity claim store.

    All writes are atomic (temp-file-and-rename) so a crash never leaves
    partial canonical state.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._entities_dir = workspace.claims_entities
        self._proposals_dir = workspace.claims_merge_proposals

    # ---- entity containers ---------------------------------------------

    def entity_path(self, entity_id: str) -> Path:
        return self._entities_dir / f"{entity_id}.yaml"

    def exists(self, entity_id: str) -> bool:
        return self.entity_path(entity_id).is_file()

    def list_entity_ids(self) -> list[str]:
        if not self._entities_dir.is_dir():
            return []
        return sorted(p.stem for p in self._entities_dir.glob("*.yaml"))

    def iter_entities(self) -> Iterator[EntityContainer]:
        for eid in self.list_entity_ids():
            yield self.load_entity(eid)

    def load_entity(self, entity_id: str) -> EntityContainer:
        path = self.entity_path(entity_id)
        if not path.is_file():
            raise ClaimStoreError(f"entity not found: {entity_id}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ClaimStoreError(f"entity file {path} must be a YAML mapping")
        try:
            return EntityContainer.from_mapping(data)
        except KeyError as exc:
            raise ClaimStoreError(f"entity file {path} missing field: {exc}") from exc

    def save_entity(self, entity: EntityContainer) -> None:
        if not ENTITY_ID_PATTERN.match(entity.entity_id):
            raise ClaimStoreError(
                f"invalid entity_id {entity.entity_id!r}: must match "
                f"{ENTITY_ID_PATTERN.pattern}"
            )
        self._entities_dir.mkdir(parents=True, exist_ok=True)
        path = self.entity_path(entity.entity_id)
        _atomic_write_yaml(path, entity.to_mapping())

    def upsert_assertion(
        self,
        entity_id: str,
        entity_type: str,
        display_name: str,
        assertion,  # type: Assertion
    ) -> EntityContainer:
        """Create-or-update an entity container with one assertion."""
        if self.exists(entity_id):
            entity = self.load_entity(entity_id)
        else:
            entity = EntityContainer(
                entity_id=entity_id,
                entity_type=entity_type,
                display_name=display_name,
            )
        existing = entity.find_assertion(assertion.claim_id)
        if existing is not None:
            # Replace.
            entity.assertions = [
                a if a.claim_id != assertion.claim_id else assertion
                for a in entity.assertions
            ]
        else:
            entity.assertions.append(assertion)
        self.save_entity(entity)
        return entity

    # ---- search helpers ------------------------------------------------

    def find_assertions_by_source(self, source_id: str) -> list[tuple[str, str]]:
        """Return (entity_id, claim_id) pairs for claims that cite ``source_id``."""
        hits: list[tuple[str, str]] = []
        for entity in self.iter_entities():
            for assertion in entity.assertions:
                if any(e.source_id == source_id for e in assertion.evidence):
                    hits.append((entity.entity_id, assertion.claim_id))
        return hits

    def find_render_targets_for_source(self, source_id: str) -> set[str]:
        """Return ``page_id`` values touched by claims citing ``source_id``."""
        page_ids: set[str] = set()
        for entity in self.iter_entities():
            for assertion in entity.assertions:
                if any(e.source_id == source_id for e in assertion.evidence):
                    for target in assertion.render_targets:
                        page_ids.add(target.page_id)
        return page_ids

    def find_entity_by_alias(self, alias_text: str) -> str | None:
        """Return the entity_id whose active aliases include ``alias_text``."""
        normalized = alias_text.strip().lower()
        for entity in self.iter_entities():
            if entity.display_name.strip().lower() == normalized:
                return entity.entity_id
            for alias in entity.aliases:
                if alias.status == "active" and alias.alias_text.strip().lower() == normalized:
                    return entity.entity_id
        return None

    # ---- merge proposals -----------------------------------------------

    def proposal_path(self, proposal_id: str) -> Path:
        return self._proposals_dir / f"{proposal_id}.yaml"

    def list_proposal_ids(self) -> list[str]:
        if not self._proposals_dir.is_dir():
            return []
        return sorted(p.stem for p in self._proposals_dir.glob("*.yaml"))

    def load_proposal(self, proposal_id: str) -> MergeProposal:
        path = self.proposal_path(proposal_id)
        if not path.is_file():
            raise ClaimStoreError(f"merge proposal not found: {proposal_id}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return MergeProposal.from_mapping(data)

    def save_proposal(self, proposal: MergeProposal) -> None:
        self._proposals_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_yaml(self.proposal_path(proposal.proposal_id), proposal.to_mapping())


def _atomic_write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

