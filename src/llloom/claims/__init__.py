"""Claim records, entity YAML containers, locators, and span verification."""

from llloom.claims.models import (
    Alias,
    Assertion,
    EntityContainer,
    Evidence,
    Locator,
    MergeProposal,
    Relation,
    RenderTarget,
)
from llloom.claims.store import ClaimStore, ClaimStoreError
from llloom.claims.verifier import (
    SpanResolutionError,
    VerifierMismatch,
    VerifierResult,
    compute_excerpt_hash,
    resolve_excerpt,
    verify_assertion,
)

__all__ = [
    "Alias",
    "Assertion",
    "ClaimStore",
    "ClaimStoreError",
    "EntityContainer",
    "Evidence",
    "Locator",
    "MergeProposal",
    "Relation",
    "RenderTarget",
    "SpanResolutionError",
    "VerifierMismatch",
    "VerifierResult",
    "compute_excerpt_hash",
    "resolve_excerpt",
    "verify_assertion",
]

