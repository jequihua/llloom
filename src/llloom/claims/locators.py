"""Per-source-class locator resolution and normalization.

Resolves a ``Locator`` against raw source text and returns the excerpt
substring. Normalizes the excerpt per locator class before hashing.

Supported locator types:

- ``markdown_prose_v1``: heading_path + paragraph_index + sentence_start/end
- ``legal_act_v1``: section_label + clause_label + paragraph_index + sentence_start/end
- ``code_v1``: 1-based inclusive line/column bounds with exact
  whitespace preservation. Line terminators (``\\n``, ``\\r\\n``,
  or lone ``\\r``) are not addressable as content columns; they
  are recovered verbatim between selected lines in a multi-line
  span. ``normalize_excerpt`` preserves whitespace byte-for-byte
  for code.
"""

from __future__ import annotations

import re

from llloom.claims.models import Locator


class SpanResolutionError(Exception):
    """Raised when a locator cannot be resolved against a source."""


# ---- public API ---------------------------------------------------------


def resolve_span(locator: Locator, source_text: str) -> str:
    """Return the verbatim excerpt text identified by ``locator``."""
    ltype = locator.locator_type
    if ltype == "markdown_prose_v1":
        return _resolve_markdown_prose(locator, source_text)
    if ltype == "legal_act_v1":
        return _resolve_legal_act(locator, source_text)
    if ltype == "code_v1":
        return _resolve_code_v1(locator, source_text)
    raise SpanResolutionError(f"unsupported locator_type: {ltype}")


def normalize_excerpt(excerpt: str, locator_type: str) -> str:
    """Normalize an excerpt for hashing.

    Markdown-prose and legal-act excerpts collapse whitespace runs to a
    single space and strip leading/trailing whitespace. Code excerpts
    preserve whitespace exactly.
    """
    if locator_type in ("markdown_prose_v1", "legal_act_v1"):
        collapsed = re.sub(r"\s+", " ", excerpt).strip()
        return collapsed
    if locator_type == "code_v1":
        return excerpt
    raise SpanResolutionError(f"unsupported locator_type: {locator_type}")


# ---- code_v1 ------------------------------------------------------------


def _resolve_code_v1(locator: Locator, source_text: str) -> str:
    """Resolve an exact line/column span.

    Line and column indices are 1-based and inclusive on both ends
    (``end_line`` / ``end_col`` name the last character that belongs to
    the span). Whitespace is preserved exactly — ``normalize_excerpt``
    does not collapse it for code excerpts. The resolver is agnostic
    to language: it slices whatever raw text the source_id points at.
    """
    start_line = locator.start_line
    start_col = locator.start_col
    end_line = locator.end_line
    end_col = locator.end_col
    if (
        start_line is None
        or start_col is None
        or end_line is None
        or end_col is None
    ):
        raise SpanResolutionError(
            "code_v1 locator requires start_line, start_col, end_line, end_col"
        )
    if start_line < 1 or start_col < 1 or end_col < 1:
        raise SpanResolutionError(
            f"code_v1 bounds must be 1-based and positive; got "
            f"start=({start_line},{start_col}) end=({end_line},{end_col})"
        )
    if end_line < start_line or (end_line == start_line and end_col < start_col):
        raise SpanResolutionError(
            f"code_v1 end precedes start: "
            f"start=({start_line},{start_col}) end=({end_line},{end_col})"
        )

    lines = source_text.splitlines(keepends=True)
    if start_line > len(lines) or end_line > len(lines):
        raise SpanResolutionError(
            f"code_v1 line out of range (source has {len(lines)} lines; "
            f"got start_line={start_line}, end_line={end_line})"
        )

    def _line_content(idx_1based: int) -> str:
        # Strip the trailing line terminator so column bounds address
        # source characters, not the terminator itself. CRLF must be
        # removed as a two-character sequence; a one-character strip
        # would leave a stray ``\r`` inside the "content" portion.
        raw = lines[idx_1based - 1]
        if raw.endswith("\r\n"):
            return raw[:-2]
        if raw.endswith(("\n", "\r")):
            return raw[:-1]
        return raw

    if start_line == end_line:
        line = _line_content(start_line)
        if start_col - 1 > len(line) or end_col > len(line):
            raise SpanResolutionError(
                f"code_v1 column out of range on line {start_line} "
                f"(line length {len(line)}; got start_col={start_col}, "
                f"end_col={end_col})"
            )
        return line[start_col - 1 : end_col]

    first = _line_content(start_line)
    last = _line_content(end_line)
    if start_col - 1 > len(first):
        raise SpanResolutionError(
            f"code_v1 start_col {start_col} out of range on line {start_line} "
            f"(line length {len(first)})"
        )
    if end_col > len(last):
        raise SpanResolutionError(
            f"code_v1 end_col {end_col} out of range on line {end_line} "
            f"(line length {len(last)})"
        )
    # Preserve the original line terminators for the middle range so
    # the excerpt round-trips byte-for-byte against the source.
    head = first[start_col - 1 :]
    middle_raw = "".join(lines[start_line : end_line - 1])
    tail = last[:end_col]
    # Recover the terminator on the start line if any (the slice above
    # dropped it via _line_content).
    start_terminator = ""
    raw_start = lines[start_line - 1]
    if raw_start.endswith("\r\n"):
        start_terminator = "\r\n"
    elif raw_start.endswith("\n"):
        start_terminator = "\n"
    elif raw_start.endswith("\r"):
        start_terminator = "\r"
    return head + start_terminator + middle_raw + tail


# ---- markdown_prose_v1 --------------------------------------------------


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _resolve_markdown_prose(locator: Locator, source_text: str) -> str:
    heading_path = locator.heading_path or []
    paragraph_index = locator.paragraph_index or 1
    sentence_start = locator.sentence_start or 1
    sentence_end = locator.sentence_end or sentence_start

    paragraphs = _paragraphs_under_heading(source_text, heading_path)
    if paragraph_index < 1 or paragraph_index > len(paragraphs):
        raise SpanResolutionError(
            f"paragraph_index {paragraph_index} out of range "
            f"(found {len(paragraphs)} paragraphs under heading_path={heading_path})"
        )
    paragraph = paragraphs[paragraph_index - 1]
    return _pick_sentences(paragraph, sentence_start, sentence_end)


def _paragraphs_under_heading(source_text: str, heading_path: list[str]) -> list[str]:
    """Return prose paragraphs under the heading path in document order.

    An empty heading_path means "paragraphs at the document root, before
    the first heading, plus paragraphs under every subsequent section" â€”
    effectively all non-heading paragraphs in document order. The first
    slice uses a strict interpretation: empty heading_path means "root
    paragraphs only, before the first heading."
    """
    lines = source_text.splitlines()
    # Walk the document tracking the current heading stack.
    heading_stack: list[tuple[int, str]] = []  # (depth, title)
    paragraphs: list[str] = []
    current_paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not current_paragraph:
            return
        text = "\n".join(current_paragraph).strip()
        if text and _path_matches(heading_stack, heading_path):
            paragraphs.append(text)
        current_paragraph.clear()

    for raw_line in lines:
        match = _HEADING_RE.match(raw_line)
        if match is not None:
            flush_paragraph()
            depth = len(match.group(1))
            title = match.group(2).strip()
            # Pop deeper or equal-depth headings.
            while heading_stack and heading_stack[-1][0] >= depth:
                heading_stack.pop()
            heading_stack.append((depth, title))
            continue

        if not raw_line.strip():
            flush_paragraph()
            continue

        current_paragraph.append(raw_line.rstrip())

    flush_paragraph()
    return paragraphs


def _path_matches(stack: list[tuple[int, str]], desired: list[str]) -> bool:
    """Return True if the current heading stack suffix matches ``desired``.

    Empty ``desired`` matches only the document root (before any heading).
    A non-empty ``desired`` of length N matches any stack whose last N
    titles equal ``desired`` element-for-element (case-sensitive, after
    trim). Suffix matching lets callers reference a deep heading path
    without enumerating the full outer hierarchy, which varies across
    fixtures.
    """
    titles = [title for _, title in stack]
    if not desired:
        return not titles
    if len(titles) < len(desired):
        return False
    return titles[-len(desired) :] == list(desired)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _pick_sentences(paragraph: str, start: int, end: int) -> str:
    """Select sentence range (1-indexed, inclusive) from ``paragraph``."""
    if start < 1 or end < start:
        raise SpanResolutionError(
            f"invalid sentence range: start={start}, end={end}"
        )
    # Collapse internal newlines into spaces for sentence detection.
    flat = re.sub(r"\s+", " ", paragraph).strip()
    sentences = _SENTENCE_SPLIT_RE.split(flat)
    if end > len(sentences):
        raise SpanResolutionError(
            f"sentence_end {end} exceeds paragraph sentence count {len(sentences)}"
        )
    selected = sentences[start - 1 : end]
    return " ".join(selected)


# ---- legal_act_v1 -------------------------------------------------------

# Legal acts use section labels like "2800." or "Section 2801." The
# first-slice implementation recognizes a section heading as a line whose
# first non-whitespace token is a numeric section label followed by a
# period, optionally inside a fenced code block (matching the observed
# fixture style in ``NCCP Act of 2003.md``). Clause labels look like
# ``(a)``, ``(b)``, etc. at the start of a line.

_SECTION_LINE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\.\s+(.*)$")
_CLAUSE_HEAD_RE = re.compile(r"^\s*(\([a-z]\))\s+(.*)$", re.IGNORECASE)


def _resolve_legal_act(locator: Locator, source_text: str) -> str:
    section_label = locator.section_label
    clause_label = locator.clause_label
    paragraph_index = locator.paragraph_index or 1
    sentence_start = locator.sentence_start or 1
    sentence_end = locator.sentence_end or sentence_start
    if not section_label:
        raise SpanResolutionError("legal_act_v1 locator requires section_label")

    # Normalize section_label to just the number (strip "Section " and "."
    # if present).
    section_number = _extract_section_number(section_label)

    section_body = _collect_section_body(source_text, section_number)
    if not section_body:
        raise SpanResolutionError(f"section {section_label!r} not found in source")

    if clause_label:
        clause_body = _collect_clause_body(section_body, clause_label)
        if not clause_body:
            raise SpanResolutionError(
                f"clause {clause_label!r} not found in section {section_label!r}"
            )
        paragraphs = _split_paragraphs(clause_body)
    else:
        paragraphs = _split_paragraphs(section_body)

    if paragraph_index < 1 or paragraph_index > len(paragraphs):
        raise SpanResolutionError(
            f"paragraph_index {paragraph_index} out of range in legal act locator "
            f"(found {len(paragraphs)} paragraphs)"
        )
    paragraph = paragraphs[paragraph_index - 1]
    return _pick_sentences(paragraph, sentence_start, sentence_end)


def _extract_section_number(label: str) -> str:
    label = label.strip()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", label)
    if not m:
        raise SpanResolutionError(f"cannot extract section number from {label!r}")
    return m.group(1)


def _collect_section_body(source_text: str, section_number: str) -> str:
    """Return the text inside the named section, up to the next section."""
    # Strip triple-backtick code fences without losing inner content. The
    # NCCP fixture wraps every clause in fenced code blocks.
    cleaned_lines = []
    for line in source_text.splitlines():
        if line.strip().startswith("```"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)

    lines = cleaned.splitlines()
    start_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        m = _SECTION_LINE_RE.match(line)
        if m is None:
            continue
        if m.group(1) == section_number and start_idx is None:
            start_idx = i
            continue
        if start_idx is not None:
            end_idx = i
            break
    if start_idx is None:
        return ""
    body_lines = lines[start_idx : end_idx if end_idx is not None else len(lines)]
    return "\n".join(body_lines)


def _collect_clause_body(section_body: str, clause_label: str) -> str:
    """Return the clause body for a label like ``(b)``."""
    want = clause_label.strip().lower()
    lines = section_body.splitlines()
    start_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        m = _CLAUSE_HEAD_RE.match(line)
        if m is None:
            continue
        if m.group(1).lower() == want and start_idx is None:
            start_idx = i
            continue
        if start_idx is not None:
            end_idx = i
            break
    if start_idx is None:
        return ""
    return "\n".join(lines[start_idx : end_idx if end_idx is not None else len(lines)])


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; drop empties."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                paragraphs.append("\n".join(current).strip())
                current = []
        else:
            current.append(line.rstrip())
    if current:
        paragraphs.append("\n".join(current).strip())
    return [p for p in paragraphs if p]

