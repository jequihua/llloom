"""Unit tests for the `code_v1` locator resolver.

`code_v1` became real in the structured-source ingest slice. It
resolves exact 1-based inclusive line/column spans against raw
source text, preserves whitespace exactly, and refuses malformed
bounds.
"""

from __future__ import annotations

import pytest

from llloom.claims.locators import (
    SpanResolutionError,
    normalize_excerpt,
    resolve_span,
)
from llloom.claims.models import Locator


SOURCE = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "class Foo:\n"
    "    def method(self):\n"
    "        x = 1\n"
    "        return x\n"
)


def _loc(**kwargs) -> Locator:
    return Locator(locator_type="code_v1", path="x.py", **kwargs)


def test_single_line_span_exact() -> None:
    # "def add(a, b):" spans line 1, cols 1..14
    loc = _loc(start_line=1, start_col=1, end_line=1, end_col=14)
    assert resolve_span(loc, SOURCE) == "def add(a, b):"


def test_single_line_span_slice_in_middle() -> None:
    # "add" on line 1 lives at cols 5..7
    loc = _loc(start_line=1, start_col=5, end_line=1, end_col=7)
    assert resolve_span(loc, SOURCE) == "add"


def test_multi_line_span_preserves_terminator_and_whitespace() -> None:
    # Lines 1..2, from col 1 of line 1 through col 16 of line 2
    # ("    return a + b" on line 2)
    loc = _loc(start_line=1, start_col=1, end_line=2, end_col=16)
    out = resolve_span(loc, SOURCE)
    assert out == "def add(a, b):\n    return a + b"


def test_normalize_excerpt_preserves_whitespace_for_code() -> None:
    text = "    x = 1\n    return x"
    assert normalize_excerpt(text, "code_v1") == text


def test_missing_bounds_refused() -> None:
    loc = _loc()
    with pytest.raises(SpanResolutionError) as exc:
        resolve_span(loc, SOURCE)
    assert "start_line" in str(exc.value) or "required" in str(exc.value)


def test_non_positive_bounds_refused() -> None:
    loc = _loc(start_line=0, start_col=1, end_line=1, end_col=1)
    with pytest.raises(SpanResolutionError):
        resolve_span(loc, SOURCE)
    loc = _loc(start_line=1, start_col=0, end_line=1, end_col=1)
    with pytest.raises(SpanResolutionError):
        resolve_span(loc, SOURCE)


def test_end_before_start_refused() -> None:
    loc = _loc(start_line=2, start_col=1, end_line=1, end_col=5)
    with pytest.raises(SpanResolutionError):
        resolve_span(loc, SOURCE)
    loc = _loc(start_line=1, start_col=10, end_line=1, end_col=5)
    with pytest.raises(SpanResolutionError):
        resolve_span(loc, SOURCE)


def test_line_out_of_range_refused() -> None:
    loc = _loc(start_line=99, start_col=1, end_line=99, end_col=1)
    with pytest.raises(SpanResolutionError) as exc:
        resolve_span(loc, SOURCE)
    assert "line out of range" in str(exc.value)


def test_column_out_of_range_refused() -> None:
    # Line 1 is "def add(a, b):" (14 chars). Column 30 is beyond.
    loc = _loc(start_line=1, start_col=1, end_line=1, end_col=30)
    with pytest.raises(SpanResolutionError) as exc:
        resolve_span(loc, SOURCE)
    assert "column out of range" in str(exc.value)


def test_class_definition_span_exact() -> None:
    # "class Foo:" is line 4, cols 1..10
    loc = _loc(start_line=4, start_col=1, end_line=4, end_col=10)
    assert resolve_span(loc, SOURCE) == "class Foo:"


def test_multi_line_span_preserves_crlf_terminator_exactly_once() -> None:
    """CRLF source text must round-trip through a multi-line span
    with exactly one ``\\r\\n`` between the two lines. A regression
    that stripped only ``\\n`` from ``_line_content`` would leak a
    stray ``\\r`` into the head slice, producing ``\\r\\r\\n``."""
    src = "abc\r\ndef\r\n"
    loc = _loc(start_line=1, start_col=1, end_line=2, end_col=3)
    assert resolve_span(loc, src) == "abc\r\ndef"


def test_crlf_terminator_not_addressable_as_content_column() -> None:
    """A CRLF-terminated line's content is the pre-terminator text;
    asking for a column that addresses the terminator itself must
    refuse rather than silently return ``\\r``."""
    src = "abc\r\ndef\r\n"
    # Line 1 content is "abc" (length 3). Column 4 is the \r of the
    # terminator; requesting it must refuse.
    loc = _loc(start_line=1, start_col=1, end_line=1, end_col=4)
    with pytest.raises(SpanResolutionError) as exc:
        resolve_span(loc, src)
    assert "column out of range" in str(exc.value)
