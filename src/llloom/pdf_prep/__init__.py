"""PDF preparation subpackage.

Optional first-party PDF working-text prep. Docling is the default provider;
it is **not** a base dependency. The conversion adapter is lazy-imported
inside `convert_with_docling`, so importing `llloom` (and every operation
that does not call into `pdf_prep`) keeps working without Docling.

See `02_analysis/docling_default_pdf_prep_milestone.md` for the milestone
contract and the future-compatibility rules around the manifest.
"""

from llloom.pdf_prep.docling import (
    DoclingArtifacts,
    DoclingNotInstalledError,
    DoclingConversionError,
    PdfPrepAdapter,
    convert_with_docling,
)
from llloom.pdf_prep.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    SELECTED_ARTIFACT_KIND,
    build_manifest,
    write_manifest,
)

__all__ = [
    "DoclingArtifacts",
    "DoclingNotInstalledError",
    "DoclingConversionError",
    "PdfPrepAdapter",
    "convert_with_docling",
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "SELECTED_ARTIFACT_KIND",
    "build_manifest",
    "write_manifest",
]
