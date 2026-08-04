"""Docling-backed PDF working-text adapter.

The adapter is the only place that imports `docling`. The import is lazy
(inside `convert_with_docling`) so a base install without
`llloom[docling]` can still import `llloom`, run the CLI help, and run
all non-Docling tests. The adapter returns plain Python strings to the
operation layer; nothing about `docling` types leaks past this module.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


@dataclass(frozen=True)
class DoclingArtifacts:
    """Plain-data result returned by a PDF prep adapter.

    `markdown` is the selected ingest artifact. `json_text` is the
    structured-export sibling (already serialized to a string by the
    adapter so the op layer never deals with adapter-specific types).
    `version` is the best-effort version string of the converter that
    produced the artifacts (`"unknown"` if not detectable).
    """

    markdown: str
    json_text: str
    version: str


class PdfPrepAdapter(Protocol):
    """Adapter callable: PDF path -> DoclingArtifacts.

    Tests inject fake adapters via this protocol so the default suite
    does not require Docling. The default adapter is
    `convert_with_docling`.
    """

    def __call__(self, pdf_path: Path) -> DoclingArtifacts: ...


class DoclingNotInstalledError(RuntimeError):
    """Raised when the optional `docling` package is not importable."""


class DoclingConversionError(RuntimeError):
    """Raised when Docling is installed but conversion failed."""


_INSTALL_HINT = 'Install with: pip install "llloom[docling]"'


def convert_with_docling(pdf_path: Path) -> DoclingArtifacts:
    """Convert ``pdf_path`` through Docling and return Markdown + JSON.

    Lazy-imports `docling` inside the function so the base install does
    not require the optional extra. Missing Docling surfaces as
    `DoclingNotInstalledError` naming the optional extra; conversion
    failures surface as `DoclingConversionError`.
    """
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DoclingNotInstalledError(
            f"docling is not installed. {_INSTALL_HINT}"
        ) from exc

    version = _detect_version()

    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        document = getattr(result, "document", None)
        if document is None:
            raise DoclingConversionError(
                "docling returned a result without a `.document` attribute"
            )
        markdown = document.export_to_markdown()
        json_data = document.export_to_dict()
    except DoclingNotInstalledError:
        raise
    except Exception as exc:
        raise DoclingConversionError(
            f"docling conversion failed for {pdf_path}: {exc}"
        ) from exc

    json_text = _json.dumps(json_data, indent=2, sort_keys=True, ensure_ascii=False)
    return DoclingArtifacts(markdown=markdown, json_text=json_text, version=version)


def _detect_version() -> str:
    """Return docling's installed version string, or ``"unknown"``."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("docling")
    except Exception:
        return "unknown"


# Type hint alias for the op layer; lets it accept a callable without
# importing the Protocol at the call site.
AdapterFn = Callable[[Path], DoclingArtifacts]
