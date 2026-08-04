"""Strict parser for model-backed extraction output.

The first model-backed extraction format is YAML with a top-level
``claims:`` array. Each entry is the structured equivalent of a
``CandidateClaim`` (a.k.a. ``SeedClaim``): everything the verifier
needs to resolve the locator, compute the canonical excerpt hash, and
persist a verified assertion.

This parser is **strict** by design:

- it accepts only the documented YAML mapping shape
- it never falls back to "best effort" extraction
- malformed YAML, the wrong top-level type, missing required fields,
  unknown locator type, or non-tuple ``render_target`` all raise
  :class:`ModelOutputError`
- empty or whitespace-only output yields zero candidates, not an error

The ingest pipeline treats ``ModelOutputError`` as a batch-atomic
refusal: no candidate from this invocation is persisted, the failure
is recorded on the operation journal entry, and the ``IngestResult``
carries a visible refusal reason. This is the safe failure mode the
spec's safety contracts demand.
"""

from __future__ import annotations

from typing import Any

import yaml

from llloom.claims.models import Locator


class ModelOutputError(Exception):
    """Raised when model output cannot be parsed into claim candidates.

    The string representation is suitable for inclusion in journal
    notes and ``IngestResult.extraction_errors``.
    """


# Required scalar fields per claim entry: each must be a non-empty string.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "entity_id",
    "entity_type",
    "display_name",
    "claim_id",
    "claim_kind",
    "claim_text",
)

# Required mapping fields per claim entry. ``locator`` is structurally
# different from the scalar required fields (it must be a mapping that
# ``Locator.from_mapping`` accepts) and is validated by its own block in
# ``_parse_one``. It is required, not optional; keeping it in a separate
# group makes the per-field validation rules legible without sliding it
# into ``_OPTIONAL_FIELDS``.
_REQUIRED_MAPPING_FIELDS: tuple[str, ...] = (
    "locator",
)

# Optional fields recognized on each claim entry. Anything else is rejected.
_OPTIONAL_FIELDS: tuple[str, ...] = (
    "render_target",
    "excerpt_hash",
    "status",
)

_KNOWN_FIELDS = frozenset(
    _REQUIRED_FIELDS + _REQUIRED_MAPPING_FIELDS + _OPTIONAL_FIELDS
)


_NARRATIVE_LOCATOR_TYPES: frozenset[str] = frozenset(
    {"markdown_prose_v1", "legal_act_v1"}
)


def parse_claim_extraction_output(
    output_text: str,
    *,
    allowed_locator_types: set[str] | frozenset[str] | None = None,
) -> list["RawCandidate"]:
    """Parse model output text into a list of structured candidates.

    Returns an empty list for empty/whitespace-only input. Raises
    :class:`ModelOutputError` for any structural problem; the caller
    is expected to treat that as batch-atomic refusal.

    ``allowed_locator_types`` restricts which ``locator.locator_type``
    values may appear in this invocation's output. The caller (ingest)
    picks the set from the schema's source-class locator: narrative
    source classes pass their single narrative locator type;
    code-backed ``claim_extract`` passes ``{"code_v1"}``. When
    ``None`` the parser falls back to the narrative-only legacy
    behavior (any narrative locator type is allowed; ``code_v1`` is
    refused) so existing callers that have not adopted the keyword
    keep the same safety story.
    """
    text = output_text or ""
    if not text.strip():
        return []

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ModelOutputError(f"YAML parse error: {exc}") from exc

    if not isinstance(data, dict):
        raise ModelOutputError(
            f"top-level value must be a mapping with a 'claims' key; "
            f"got {type(data).__name__}"
        )

    claims_raw = data.get("claims")
    if claims_raw is None:
        raise ModelOutputError("missing required top-level key 'claims'")
    if not isinstance(claims_raw, list):
        raise ModelOutputError(
            f"'claims' must be a list; got {type(claims_raw).__name__}"
        )

    if allowed_locator_types is None:
        effective_allowed: frozenset[str] = _NARRATIVE_LOCATOR_TYPES
    else:
        effective_allowed = frozenset(allowed_locator_types)
        if not effective_allowed:
            raise ModelOutputError(
                "allowed_locator_types must not be empty; "
                "ingest must name at least one acceptable locator type"
            )

    candidates: list[RawCandidate] = []
    for index, entry in enumerate(claims_raw):
        candidates.append(_parse_one(entry, index, effective_allowed))
    return candidates


# ---- internal candidate shape -------------------------------------------


class RawCandidate:
    """Structured candidate parsed from model output.

    Mirrors the public :class:`llloom.ops.ingest.SeedClaim` field set
    one-to-one. We use a separate class here so the parser can be
    imported without a cycle into ``ops.ingest``; the ingest module
    converts ``RawCandidate`` instances into ``SeedClaim`` instances.
    """

    __slots__ = (
        "entity_id",
        "entity_type",
        "display_name",
        "claim_id",
        "claim_kind",
        "claim_text",
        "locator",
        "render_target",
        "excerpt_hash",
        "status",
    )

    def __init__(
        self,
        *,
        entity_id: str,
        entity_type: str,
        display_name: str,
        claim_id: str,
        claim_kind: str,
        claim_text: str,
        locator: Locator,
        render_target: tuple[str, str] | None,
        excerpt_hash: str | None,
        status: str,
    ) -> None:
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.display_name = display_name
        self.claim_id = claim_id
        self.claim_kind = claim_kind
        self.claim_text = claim_text
        self.locator = locator
        self.render_target = render_target
        self.excerpt_hash = excerpt_hash
        self.status = status


def _parse_one(
    entry: Any, index: int, allowed_locator_types: frozenset[str]
) -> RawCandidate:
    if not isinstance(entry, dict):
        raise ModelOutputError(
            f"claims[{index}] must be a mapping; got {type(entry).__name__}"
        )

    unknown = sorted(set(entry) - _KNOWN_FIELDS)
    if unknown:
        raise ModelOutputError(
            f"claims[{index}] has unknown fields: {unknown}; "
            f"allowed: {sorted(_KNOWN_FIELDS)}"
        )

    for field in _REQUIRED_FIELDS:
        if field not in entry:
            raise ModelOutputError(
                f"claims[{index}] missing required field {field!r}"
            )
        value = entry[field]
        if not isinstance(value, str) or not value.strip():
            raise ModelOutputError(
                f"claims[{index}].{field} must be a non-empty string"
            )

    locator_raw = entry.get("locator")
    if not isinstance(locator_raw, dict):
        raise ModelOutputError(
            f"claims[{index}] missing or invalid 'locator': must be a mapping"
        )
    if "locator_type" not in locator_raw:
        raise ModelOutputError(
            f"claims[{index}].locator missing required key 'locator_type'"
        )
    locator_type = locator_raw.get("locator_type")
    if locator_type not in allowed_locator_types:
        raise ModelOutputError(
            f"claims[{index}].locator: locator_type {locator_type!r} is not "
            f"allowed for this ingest; allowed: {sorted(allowed_locator_types)}"
        )
    try:
        locator = Locator.from_mapping(dict(locator_raw))
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelOutputError(
            f"claims[{index}].locator failed to parse: {exc}"
        ) from exc

    render_target_raw = entry.get("render_target")
    render_target: tuple[str, str] | None = None
    if render_target_raw is not None:
        rt = _parse_render_target(render_target_raw, index)
        render_target = rt

    excerpt_hash_raw = entry.get("excerpt_hash")
    if excerpt_hash_raw is not None and not isinstance(excerpt_hash_raw, str):
        raise ModelOutputError(
            f"claims[{index}].excerpt_hash must be a string when present"
        )

    status_raw = entry.get("status", "draft")
    if not isinstance(status_raw, str) or not status_raw.strip():
        raise ModelOutputError(
            f"claims[{index}].status must be a non-empty string when present"
        )

    return RawCandidate(
        entity_id=entry["entity_id"],
        entity_type=entry["entity_type"],
        display_name=entry["display_name"],
        claim_id=entry["claim_id"],
        claim_kind=entry["claim_kind"],
        claim_text=entry["claim_text"],
        locator=locator,
        render_target=render_target,
        excerpt_hash=excerpt_hash_raw,
        status=status_raw,
    )


def _parse_render_target(value: Any, index: int) -> tuple[str, str]:
    """Accept ``[page_id, block_id]`` (YAML list) or
    ``{page_id: ..., block_id: ...}`` (mapping). Both are common YAML
    encodings of the ``(page_id, block_id)`` tuple."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or not all(isinstance(v, str) for v in value):
            raise ModelOutputError(
                f"claims[{index}].render_target list must be [page_id, block_id]"
            )
        return (str(value[0]), str(value[1]))
    if isinstance(value, dict):
        page = value.get("page_id")
        block = value.get("block_id")
        if not isinstance(page, str) or not isinstance(block, str):
            raise ModelOutputError(
                f"claims[{index}].render_target mapping needs string "
                f"'page_id' and 'block_id'"
            )
        return (page, block)
    raise ModelOutputError(
        f"claims[{index}].render_target must be a list of [page_id, block_id] "
        f"or a mapping with 'page_id' and 'block_id'"
    )
