"""Unit tests for the PDF-prep manifest builder + writer.

Pins the `pdf_prep_manifest_v1` shape from
`02_analysis/docling_default_pdf_prep_milestone.md`. The manifest must
stay provider-neutral: future-pipeline component slots
(`pymupdf`, `grobid`, `pdfplumber`, `nougat`) always appear, marked
`not_run` until a companion producer actually exercises them.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from llloom.pdf_prep.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    SELECTED_ARTIFACT_KIND,
    build_manifest,
    sha256_hex,
    sha256_of_file,
    write_manifest,
)


def test_build_manifest_succeeded_shape() -> None:
    manifest = build_manifest(
        prep_id="example",
        status="succeeded",
        source_pdf_workspace_path="raw/sources/papers/example.pdf",
        source_pdf_sha256="sha256:" + "a" * 64,
        artifacts=[
            {
                "path": "raw/derived/pdf/example/docling.md",
                "kind": SELECTED_ARTIFACT_KIND,
                "sha256": "sha256:" + "b" * 64,
            },
            {
                "path": "raw/derived/pdf/example/docling.json",
                "kind": "docling_json",
                "sha256": "sha256:" + "c" * 64,
            },
        ],
        selected_ingest_artifact={
            "path": "raw/derived/pdf/example/docling.md",
            "kind": SELECTED_ARTIFACT_KIND,
            "sha256": "sha256:" + "b" * 64,
        },
        docling_status="succeeded",
        docling_version="2.13.1",
    )
    assert manifest["version"] == MANIFEST_VERSION
    assert manifest["prep_id"] == "example"
    assert manifest["provider"] == "docling_default"
    assert manifest["status"] == "succeeded"
    assert manifest["source_pdf"] == {
        "path": "raw/sources/papers/example.pdf",
        "sha256": "sha256:" + "a" * 64,
    }
    assert manifest["selected_ingest_artifact"]["path"].endswith("docling.md")
    assert manifest["selected_ingest_artifact"]["kind"] == SELECTED_ARTIFACT_KIND


def test_manifest_future_components_marked_not_run() -> None:
    manifest = build_manifest(
        prep_id="example",
        status="succeeded",
        source_pdf_workspace_path="raw/sources/example.pdf",
        source_pdf_sha256="sha256:" + "0" * 64,
        artifacts=[],
        selected_ingest_artifact=None,
        docling_status="succeeded",
        docling_version="2.13.1",
    )
    components = manifest["components"]
    assert components["docling"]["status"] == "succeeded"
    assert components["docling"]["version"] == "2.13.1"
    for future in ("pymupdf", "grobid", "pdfplumber", "nougat"):
        assert future in components, f"missing future component slot: {future}"
        assert components[future] == {"status": "not_run"}


def test_manifest_failed_status_drops_selected_artifact() -> None:
    manifest = build_manifest(
        prep_id="example",
        status="failed",
        source_pdf_workspace_path="raw/sources/example.pdf",
        source_pdf_sha256="sha256:" + "1" * 64,
        artifacts=[],
        selected_ingest_artifact=None,
        docling_status="failed",
        docling_version="unknown",
    )
    assert manifest["status"] == "failed"
    assert manifest["selected_ingest_artifact"] is None
    assert manifest["components"]["docling"]["status"] == "failed"


def test_write_manifest_is_atomic_and_yaml_roundtrips(tmp_path: Path) -> None:
    manifest = build_manifest(
        prep_id="example",
        status="succeeded",
        source_pdf_workspace_path="raw/sources/example.pdf",
        source_pdf_sha256="sha256:" + "2" * 64,
        artifacts=[
            {
                "path": "raw/derived/pdf/example/docling.md",
                "kind": SELECTED_ARTIFACT_KIND,
                "sha256": "sha256:" + "3" * 64,
            }
        ],
        selected_ingest_artifact={
            "path": "raw/derived/pdf/example/docling.md",
            "kind": SELECTED_ARTIFACT_KIND,
            "sha256": "sha256:" + "3" * 64,
        },
        docling_status="succeeded",
        docling_version="2.13.1",
    )
    target = tmp_path / "bundle" / MANIFEST_FILENAME
    write_manifest(target, manifest)
    assert target.is_file()
    assert not target.with_suffix(target.suffix + ".tmp").exists()
    parsed = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert parsed == manifest


def test_sha256_helpers_format(tmp_path: Path) -> None:
    assert sha256_hex(b"") == "sha256:" + (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.7\n%not-real\n")
    digest = sha256_of_file(pdf)
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
