"""Seed apply update reports + excerpt-equality helper.

Slice 076 added durable per-apply audit evidence under
``state/reports/updates/<op_id>.yaml``. Each real, mutating
``llloom seed apply`` writes one YAML report on success: it captures
the planned and created claims, source / manifest hashes, before /
after workspace counts, bounded excerpt previews, and provenance
fields proving the apply path never invoked a model. Dry-run writes
nothing. Validation refusals before persistence write nothing.

The module also exposes the deterministic
:func:`check_seed_excerpt_equality` helper. Manifests may opt a
single-sentence claim into ``excerpt_equality: exact_one_sentence``;
the helper normalizes both the manifest ``claim_text`` and the
locator-resolved excerpt using the same whitespace-collapse rule the
markdown-prose verifier uses, and refuses the source batch atomically
on mismatch. The helper is pure (no I/O, no provider call) and is
re-used by both the seed apply code and the report writer.

See ``04_specification/seed_manifest_v1.md`` for the manifest field
and ``04_specification/storage_and_state_model.md`` for the report
contract.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from llloom.workspace.layout import Workspace


SEED_UPDATE_REPORT_VERSION = "seed_update_report_v1"

# Excerpt previews and refusal diagnostics use the same bound the
# verifier uses for its mismatch previews. Small enough that previews
# never accidentally carry large source bodies into reports or stdout.
EXCERPT_PREVIEW_MAX_CHARS = 240

EXCERPT_EQUALITY_MODES: frozenset[str] = frozenset({"none", "exact_one_sentence"})


@dataclass(frozen=True)
class SeedExcerptCheck:
    """One excerpt-equality decision for a single seed-manifest claim.

    Produced by :func:`check_seed_excerpt_equality`. The ``mode`` is
    the manifest's declared check mode (``none`` or
    ``exact_one_sentence``). ``matched`` is the boolean outcome.
    ``excerpt_hash`` is the SHA-256 of the *normalized* resolved
    excerpt; ``excerpt_preview`` is a bounded preview of that
    normalized text. ``message`` is non-None only on a mismatch and
    carries the actionable diagnostic the seed apply path surfaces
    via ``SeedManifestResult.refusal_reason``.
    """

    claim_id: str
    mode: str
    matched: bool
    excerpt_hash: str
    excerpt_preview: str
    message: str | None = None


def _normalize_one_sentence(text: str) -> str:
    """Whitespace-collapse normalization for one-sentence prose.

    Identical rule to ``llloom.claims.locators.normalize_excerpt``
    for ``markdown_prose_v1`` — collapse all whitespace runs to a
    single space and strip leading / trailing whitespace. The
    seed report module owns this copy because the excerpt-equality
    check applies the same rule independent of locator class.
    """
    return re.sub(r"\s+", " ", text).strip()


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _preview(text: str, max_chars: int = EXCERPT_PREVIEW_MAX_CHARS) -> str:
    """Bounded preview of ``text``. Length is always ``<= max_chars``."""
    if max_chars < 3:
        raise ValueError("max_chars must be at least 3")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def check_seed_excerpt_equality(
    *,
    claim_id: str,
    claim_text: str,
    resolved_excerpt: str,
    mode: str | None,
) -> SeedExcerptCheck:
    """Decide whether a seed manifest claim's text matches the
    locator-resolved excerpt under ``mode``.

    Modes:

    - ``None`` / ``"none"`` — no extra equality requirement. The
      verifier's ``excerpt_hash`` resolution still runs separately;
      this helper returns ``matched=True`` with the normalized
      excerpt's preview + hash so the report still surfaces the
      resolved excerpt.
    - ``"exact_one_sentence"`` — normalize both ``claim_text`` and
      ``resolved_excerpt`` with the whitespace-collapse rule used
      for markdown-prose hashing, then require exact string
      equality. Mismatches set ``matched=False`` and carry a
      bounded ``message`` naming both previews.

    The helper is pure: no I/O, no provider call, no logging. It
    is safe to call inside lockless preflight before the seed
    apply opens any ``operation(...)`` context.
    """
    effective_mode = mode if mode is not None else "none"
    if effective_mode not in EXCERPT_EQUALITY_MODES:
        raise ValueError(
            f"unsupported excerpt_equality mode {effective_mode!r}; "
            f"allowed: {sorted(EXCERPT_EQUALITY_MODES)}"
        )

    normalized_excerpt = _normalize_one_sentence(resolved_excerpt)
    excerpt_hash = _hash_text(normalized_excerpt)
    excerpt_preview = _preview(normalized_excerpt)

    if effective_mode == "none":
        return SeedExcerptCheck(
            claim_id=claim_id,
            mode="none",
            matched=True,
            excerpt_hash=excerpt_hash,
            excerpt_preview=excerpt_preview,
        )

    normalized_text = _normalize_one_sentence(claim_text)
    matched = normalized_text == normalized_excerpt
    message: str | None = None
    if not matched:
        message = (
            f"claim {claim_id!r}: excerpt_equality=exact_one_sentence "
            f"failed; manifest claim_text does not equal locator-resolved "
            f"excerpt. claim_text preview={_preview(normalized_text)!r}; "
            f"excerpt preview={excerpt_preview!r}"
        )
    return SeedExcerptCheck(
        claim_id=claim_id,
        mode="exact_one_sentence",
        matched=matched,
        excerpt_hash=excerpt_hash,
        excerpt_preview=excerpt_preview,
        message=message,
    )


def iso_now() -> str:
    """UTC iso-8601 timestamp with second precision. Stable for tests."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_file_sha256(path: Path) -> str:
    """SHA-256 of ``path``, prefixed ``sha256:``. Matches the source
    registry's hash format so the report's manifest / source hashes
    are directly comparable to registry records.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def report_path(workspace: Workspace, op_id: str) -> Path:
    """Workspace report path for ``op_id``."""
    return workspace.state_reports_updates / f"{op_id}.yaml"


def write_seed_update_report(
    workspace: Workspace,
    *,
    op_id: str,
    payload: dict[str, Any],
) -> Path:
    """Persist ``payload`` as YAML under ``state/reports/updates/<op_id>.yaml``.

    Writes atomically via temp-file-and-rename so a crash never leaves
    a partially-written report. Returns the absolute path written.
    """
    target = report_path(workspace, op_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    return target


def excerpt_check_to_mapping(check: SeedExcerptCheck) -> dict[str, Any]:
    """Serialize a :class:`SeedExcerptCheck` for inclusion in the
    YAML report. ``message`` is dropped when None to keep the report
    tidy.
    """
    data = asdict(check)
    if data.get("message") is None:
        data.pop("message", None)
    return data
