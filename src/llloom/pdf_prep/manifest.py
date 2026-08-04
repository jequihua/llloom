"""PDF-prep manifest builder + atomic writer.

Implements the `pdf_prep_manifest_v1` shape from
`02_analysis/docling_default_pdf_prep_milestone.md`. The manifest is
provider-neutral: it always carries `components` slots for the future
PyMuPDF + GROBID + pdfplumber + Nougat pipeline (`not_run` until those
components are actually used), so a future companion package can
produce the same contract with richer artifacts while `llloom` still
ingests exactly one selected frozen text artifact.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Iterable

import yaml

MANIFEST_FILENAME = "pdf_prep_manifest.yaml"
MANIFEST_VERSION = "pdf_prep_manifest_v1"
SELECTED_ARTIFACT_KIND = "docling_markdown"
PROVIDER_DOCLING_DEFAULT = "docling_default"

# Future-pipeline component slots reserved on the manifest. The Docling
# default workflow populates `docling`; everything else stays `not_run`.
# Future companion producers will set the others as they run.
_DEFAULT_COMPONENTS: tuple[str, ...] = (
    "docling",
    "pymupdf",
    "grobid",
    "pdfplumber",
    "nougat",
)


def sha256_hex(data: bytes) -> str:
    """Return the SHA-256 of ``data`` as ``sha256:<lowercase hex>``."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_of_file(path: Path) -> str:
    """Stream-hash a file and return ``sha256:<hex>``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _iso_now_utc() -> str:
    """ISO-8601 UTC timestamp matching the rest of the package."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _llloom_version() -> str:
    try:
        return _pkg_version("llloom")
    except PackageNotFoundError:
        return "unknown"


def build_manifest(
    *,
    prep_id: str,
    status: str,
    source_pdf_workspace_path: str,
    source_pdf_sha256: str,
    artifacts: Iterable[dict],
    selected_ingest_artifact: dict | None,
    docling_status: str,
    docling_version: str,
    provider: str = PROVIDER_DOCLING_DEFAULT,
) -> dict:
    """Build the provider-neutral manifest dict.

    `artifacts` items are ``{"path", "kind", "sha256"}``. The
    `selected_ingest_artifact` is the single frozen text artifact that
    a normal `llloom ingest` will register; on failure it should be
    ``None`` and `status` should be ``"failed"``.
    """
    components: dict[str, dict] = {}
    for name in _DEFAULT_COMPONENTS:
        if name == "docling":
            slot: dict[str, str] = {"status": docling_status}
            if docling_status != "not_run":
                slot["version"] = docling_version or "unknown"
            components[name] = slot
        else:
            components[name] = {"status": "not_run"}

    manifest: dict = {
        "version": MANIFEST_VERSION,
        "prep_id": prep_id,
        "provider": provider,
        "status": status,
        "source_pdf": {
            "path": source_pdf_workspace_path,
            "sha256": source_pdf_sha256,
        },
        "components": components,
        "artifacts": [dict(a) for a in artifacts],
        "selected_ingest_artifact": (
            dict(selected_ingest_artifact)
            if selected_ingest_artifact is not None
            else None
        ),
        "tooling": {
            "generated_at": _iso_now_utc(),
            "llloom_version": _llloom_version(),
            "docling_version": docling_version or "unknown",
        },
    }
    return manifest


def write_manifest(manifest_path: Path, manifest: dict) -> None:
    """Atomically write ``manifest`` to ``manifest_path`` as YAML.

    Uses the same temp-file-and-rename idiom as the rest of the
    package's atomic writers.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(manifest_path)
