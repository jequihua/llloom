"""Page parsing, region management, and claim-block rendering (variant B)."""

from llloom.pages.regions import (
    CLAIM_BLOCK_START_RE,
    CLAIM_BLOCK_END_RE,
    COMMENTARY_START_RE,
    COMMENTARY_END_RE,
    PageParseError,
    ParsedPage,
    parse_page,
    replace_claim_block,
)
from llloom.pages.render import (
    RenderError,
    RenderResult,
    compute_render_fingerprint,
    render_claim_block,
    render_page_file,
    resolve_page_path,
)

__all__ = [
    "CLAIM_BLOCK_END_RE",
    "CLAIM_BLOCK_START_RE",
    "COMMENTARY_END_RE",
    "COMMENTARY_START_RE",
    "PageParseError",
    "ParsedPage",
    "RenderError",
    "RenderResult",
    "compute_render_fingerprint",
    "parse_page",
    "render_claim_block",
    "render_page_file",
    "replace_claim_block",
    "resolve_page_path",
]

