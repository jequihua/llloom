"""Deterministic structured-source extraction for the
``structure_extract`` ingest policy.

Structure reports are derived state under ``state/structure/``: they
are rebuildable, non-canonical, and deletable without losing any
canonical knowledge. Reports contain structure only — key paths,
symbols, kinds, and ``code_v1`` locators — and deliberately omit raw
scalar values, comments, docstrings, source lines, and code bodies.

The YAML extractor runs in the base install using stdlib + PyYAML.
The Python extractor is gated behind the optional ``llloom[structured]``
extra and lazy-imports tree-sitter on first use.
"""

from llloom.structured.extract import (
    StructureExtractError,
    StructureItem,
    StructureReport,
    SUPPORTED_SOURCE_CLASSES,
    extract_structure,
    write_structure_report,
)

__all__ = [
    "SUPPORTED_SOURCE_CLASSES",
    "StructureExtractError",
    "StructureItem",
    "StructureReport",
    "extract_structure",
    "write_structure_report",
]
