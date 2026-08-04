"""Page region parser for variant-(B) pages.

The frozen marker syntax is HTML-comment fences with stable region IDs.
See ``04_specification/storage_and_state_model.md`` Â§"Page region marker
syntax".

Rules enforced here:

- exactly one ``<!-- llloom:claim-block id=... -->`` / ``<!-- /llloom:claim-block -->`` pair
- exactly one ``<!-- llloom:commentary id=... -->`` / ``<!-- /llloom:commentary -->`` pair
- region IDs are required
- missing, duplicate, or malformed markers are hard parse failures
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml


CLAIM_BLOCK_START_RE = re.compile(
    r"<!--\s*llloom:claim-block\s+id=([A-Za-z0-9._-]+)(?:\s+[^>]*?)?\s*-->"
)
CLAIM_BLOCK_END_RE = re.compile(r"<!--\s*/llloom:claim-block\s*-->")
COMMENTARY_START_RE = re.compile(
    r"<!--\s*llloom:commentary\s+id=([A-Za-z0-9._-]+)(?:\s+[^>]*?)?\s*-->"
)
COMMENTARY_END_RE = re.compile(r"<!--\s*/llloom:commentary\s*-->")

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


class PageParseError(Exception):
    """Raised for malformed page markers."""


@dataclass
class Region:
    id: str
    inner_start: int  # byte offset immediately after end of start marker line
    inner_end: int  # byte offset of first character of end marker
    start_marker_start: int
    end_marker_end: int

    @property
    def inner_text(self) -> str:  # pragma: no cover - helper
        raise NotImplementedError


@dataclass
class ParsedPage:
    """Result of parsing a variant-(B) page."""

    raw_text: str
    frontmatter: dict
    frontmatter_text: str
    claim_block_id: str
    claim_block_inner: str  # bytes between markers (excludes marker lines)
    claim_block_outer_start: int
    claim_block_outer_end: int
    commentary_id: str
    commentary_inner: str
    commentary_outer_start: int
    commentary_outer_end: int

    def with_claim_block(self, new_inner: str) -> str:
        """Return the full page text with the claim-block inner replaced."""
        return replace_claim_block(self, new_inner)


def parse_page(page_text: str) -> ParsedPage:
    """Parse a variant-(B) page text.

    Raises PageParseError for any violation of the marker contract.
    """
    frontmatter, frontmatter_text, body_offset = _parse_frontmatter(page_text)

    claim_starts = list(CLAIM_BLOCK_START_RE.finditer(page_text))
    claim_ends = list(CLAIM_BLOCK_END_RE.finditer(page_text))
    comm_starts = list(COMMENTARY_START_RE.finditer(page_text))
    comm_ends = list(COMMENTARY_END_RE.finditer(page_text))

    if len(claim_starts) != 1 or len(claim_ends) != 1:
        raise PageParseError(
            f"page must contain exactly one claim-block pair; "
            f"found {len(claim_starts)} start markers and {len(claim_ends)} end markers"
        )
    if len(comm_starts) != 1 or len(comm_ends) != 1:
        raise PageParseError(
            f"page must contain exactly one commentary pair; "
            f"found {len(comm_starts)} start markers and {len(comm_ends)} end markers"
        )

    cb_start_m = claim_starts[0]
    cb_end_m = claim_ends[0]
    cm_start_m = comm_starts[0]
    cm_end_m = comm_ends[0]

    if cb_end_m.start() <= cb_start_m.end():
        raise PageParseError("claim-block end marker precedes start marker")
    if cm_end_m.start() <= cm_start_m.end():
        raise PageParseError("commentary end marker precedes start marker")

    # Regions must be non-overlapping.
    regions = sorted(
        [
            ("claim-block", cb_start_m.start(), cb_end_m.end()),
            ("commentary", cm_start_m.start(), cm_end_m.end()),
        ],
        key=lambda t: t[1],
    )
    if regions[0][2] > regions[1][1]:
        raise PageParseError("claim-block and commentary regions overlap")

    claim_block_inner = _extract_inner(page_text, cb_start_m.end(), cb_end_m.start())
    commentary_inner = _extract_inner(page_text, cm_start_m.end(), cm_end_m.start())

    return ParsedPage(
        raw_text=page_text,
        frontmatter=frontmatter,
        frontmatter_text=frontmatter_text,
        claim_block_id=cb_start_m.group(1),
        claim_block_inner=claim_block_inner,
        claim_block_outer_start=cb_start_m.start(),
        claim_block_outer_end=cb_end_m.end(),
        commentary_id=cm_start_m.group(1),
        commentary_inner=commentary_inner,
        commentary_outer_start=cm_start_m.start(),
        commentary_outer_end=cm_end_m.end(),
    )


def replace_claim_block(parsed: ParsedPage, new_inner: str) -> str:
    """Return full page text with the claim-block inner replaced.

    Preserves:

    - frontmatter
    - commentary bytes exactly
    - everything outside the claim-block markers
    """
    page = parsed.raw_text
    cb_start_marker_end = _find_line_end(page, parsed.claim_block_outer_start)
    cb_end_marker_start = _find_line_start(page, parsed.claim_block_outer_end)

    # Compose: everything up to and including the start marker line,
    # then the new inner (with surrounding newlines), then the end marker
    # line, then the rest.
    before = page[: cb_start_marker_end + 1]
    after = page[cb_end_marker_start:]
    normalized_inner = new_inner.strip("\n")
    if normalized_inner:
        middle = "\n" + normalized_inner + "\n"
    else:
        middle = "\n"
    return before + middle + after


# ---- internal helpers ---------------------------------------------------


def _parse_frontmatter(text: str) -> tuple[dict, str, int]:
    match = FRONTMATTER_RE.match(text)
    if match is None:
        return {}, "", 0
    fm_text = match.group(1)
    try:
        data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:
        raise PageParseError(f"malformed frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        raise PageParseError("frontmatter must be a YAML mapping")
    return data, fm_text, match.end()


def _extract_inner(text: str, start: int, end: int) -> str:
    """Extract the text between the start marker's end and end marker's start.

    Trims exactly one surrounding newline on each side if present.
    """
    chunk = text[start:end]
    if chunk.startswith("\n"):
        chunk = chunk[1:]
    if chunk.endswith("\n"):
        chunk = chunk[:-1]
    return chunk


def _find_line_end(text: str, offset: int) -> int:
    """Return the offset of the ``\\n`` ending the line that contains ``offset``.

    If ``offset`` falls on the last line without a trailing newline, returns
    len(text) - 1.
    """
    nl = text.find("\n", offset)
    if nl == -1:
        return len(text) - 1
    return nl


def _find_line_start(text: str, offset: int) -> int:
    """Return the offset of the beginning of the line that contains ``offset``.

    If the character at ``offset-1`` is ``\\n``, returns ``offset``.
    """
    if offset == 0:
        return 0
    nl = text.rfind("\n", 0, offset)
    if nl == -1:
        return 0
    return nl + 1

