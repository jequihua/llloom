"""Unit tests for ``llloom.ops.prepare_pdf``.

Exercises the op layer through a fake adapter so the default suite
does not require the optional `llloom[docling]` extra. Covers:

- success path: bundle + manifest + selected artifact under
  ``raw/derived/pdf/<prep_id>/``;
- missing-optional-dependency path: refusal naming
  ``llloom[docling]`` with no successful selected artifact;
- conversion-failure path: failed-status manifest, no selected
  artifact, refusal reason surfaced on the result;
- overwrite refusal: existing bundle without ``--overwrite``
  refuses; explicit ``--overwrite`` succeeds without touching
  sibling files outside the bundle.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from llloom.ops.prepare_pdf import prepare_pdf
from llloom.pdf_prep.docling import (
    DoclingArtifacts,
    DoclingConversionError,
    DoclingNotInstalledError,
)
from llloom.pdf_prep.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    SELECTED_ARTIFACT_KIND,
)
from llloom.workspace.layout import Workspace


_FAKE_MARKDOWN = "# Example\n\nFake docling markdown body.\n"
_FAKE_JSON = '{"name": "fake", "pages": 1}'


def _fake_adapter(pdf_path: Path) -> DoclingArtifacts:
    assert pdf_path.is_file()
    return DoclingArtifacts(
        markdown=_FAKE_MARKDOWN,
        json_text=_FAKE_JSON,
        version="2.13.1-fake",
    )


def _write_pdf_under_workspace(ws: Workspace, name: str = "example.pdf") -> Path:
    pdf = ws.raw_sources / name
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.7\n%not-a-real-pdf\n")
    return pdf


def test_prepare_pdf_success_with_fake_adapter(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    pdf = _write_pdf_under_workspace(ws)

    result = prepare_pdf(ws, pdf_path=pdf, adapter=_fake_adapter)

    assert result.succeeded, result.refusal_reason
    assert result.status == "succeeded"
    assert result.prep_id == "example"
    assert result.bundle_dir == "raw/derived/pdf/example"
    assert result.manifest_path == "raw/derived/pdf/example/pdf_prep_manifest.yaml"
    assert result.selected_artifact is not None
    assert result.selected_artifact.kind == SELECTED_ARTIFACT_KIND
    assert result.selected_artifact.path == "raw/derived/pdf/example/docling.md"
    assert result.components["docling"] == "succeeded"
    assert result.op_id.startswith("op.prepare_pdf.")

    bundle = ws.root / "raw" / "derived" / "pdf" / "example"
    md = bundle / "docling.md"
    js = bundle / "docling.json"
    mf = bundle / MANIFEST_FILENAME
    assert md.read_text(encoding="utf-8") == _FAKE_MARKDOWN
    assert js.read_text(encoding="utf-8") == _FAKE_JSON
    assert mf.is_file()
    # No temp files left behind.
    assert not any(p.suffix == ".tmp" for p in bundle.iterdir())

    parsed = yaml.safe_load(mf.read_text(encoding="utf-8"))
    assert parsed["version"] == MANIFEST_VERSION
    assert parsed["status"] == "succeeded"
    assert parsed["provider"] == "docling_default"
    assert parsed["selected_ingest_artifact"]["path"] == result.selected_artifact.path
    assert parsed["components"]["docling"] == {
        "status": "succeeded",
        "version": "2.13.1-fake",
    }
    for future in ("pymupdf", "grobid", "pdfplumber", "nougat"):
        assert parsed["components"][future] == {"status": "not_run"}


def test_prepare_pdf_missing_optional_dependency(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    pdf = _write_pdf_under_workspace(ws)

    def missing_adapter(_: Path) -> DoclingArtifacts:
        raise DoclingNotInstalledError(
            'docling is not installed. Install with: pip install "llloom[docling]"'
        )

    result = prepare_pdf(ws, pdf_path=pdf, adapter=missing_adapter)

    assert not result.succeeded
    assert result.status == "refused"
    assert result.selected_artifact is None
    assert result.refusal_reason is not None
    assert 'llloom[docling]' in result.refusal_reason
    # No bundle artifacts written on missing-extra refusal.
    bundle = ws.root / "raw" / "derived" / "pdf" / "example"
    assert not (bundle / "docling.md").exists()
    assert not (bundle / "docling.json").exists()
    assert not (bundle / MANIFEST_FILENAME).exists()


def test_prepare_pdf_conversion_failure_writes_failed_manifest(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    pdf = _write_pdf_under_workspace(ws)

    def boom_adapter(_: Path) -> DoclingArtifacts:
        raise DoclingConversionError("simulated docling crash")

    result = prepare_pdf(ws, pdf_path=pdf, adapter=boom_adapter)

    assert not result.succeeded
    assert result.status == "failed"
    assert result.selected_artifact is None
    assert result.refusal_reason is not None
    assert "simulated docling crash" in result.refusal_reason
    bundle = ws.root / "raw" / "derived" / "pdf" / "example"
    mf = bundle / MANIFEST_FILENAME
    assert mf.is_file(), "failed manifest must be written for audit trail"
    parsed = yaml.safe_load(mf.read_text(encoding="utf-8"))
    assert parsed["status"] == "failed"
    assert parsed["selected_ingest_artifact"] is None
    assert parsed["components"]["docling"]["status"] == "failed"
    # No selected artifact files on failure.
    assert not (bundle / "docling.md").exists()
    assert not (bundle / "docling.json").exists()


def test_prepare_pdf_refuses_existing_bundle_without_overwrite(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    pdf = _write_pdf_under_workspace(ws)

    first = prepare_pdf(ws, pdf_path=pdf, adapter=_fake_adapter)
    assert first.succeeded

    second = prepare_pdf(ws, pdf_path=pdf, adapter=_fake_adapter)
    assert not second.succeeded
    assert second.status == "refused"
    assert second.refusal_reason is not None
    assert "already exists" in second.refusal_reason
    assert "--overwrite" in second.refusal_reason


def test_prepare_pdf_overwrite_replaces_only_bundle_files(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    pdf = _write_pdf_under_workspace(ws)

    sibling_root = ws.root / "raw" / "derived" / "pdf"
    first = prepare_pdf(ws, pdf_path=pdf, adapter=_fake_adapter)
    assert first.succeeded

    # Drop a sibling bundle for a different prep id and a sibling file
    # under raw/derived/pdf/ so we can assert overwrite leaves them
    # alone.
    sibling_bundle = sibling_root / "other-prep"
    sibling_bundle.mkdir(parents=True)
    (sibling_bundle / "docling.md").write_text("# Other", encoding="utf-8")
    sibling_file = sibling_root / "notes.txt"
    sibling_file.write_text("unrelated", encoding="utf-8")

    def updated_adapter(_: Path) -> DoclingArtifacts:
        return DoclingArtifacts(
            markdown="# Example (updated)\n",
            json_text='{"name": "fake-updated"}',
            version="2.13.1-fake",
        )

    second = prepare_pdf(
        ws, pdf_path=pdf, adapter=updated_adapter, overwrite=True
    )
    assert second.succeeded
    bundle = ws.root / "raw" / "derived" / "pdf" / "example"
    assert (
        (bundle / "docling.md").read_text(encoding="utf-8")
        == "# Example (updated)\n"
    )
    # Sibling bundle and sibling file untouched.
    assert (sibling_bundle / "docling.md").read_text(encoding="utf-8") == "# Other"
    assert sibling_file.read_text(encoding="utf-8") == "unrelated"


def test_prepare_pdf_explicit_prep_id_and_output_dir(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    pdf = ws.raw_sources / "papers" / "weirdly named v1.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.7\n")

    result = prepare_pdf(
        ws,
        pdf_path=pdf,
        prep_id="paper-1",
        output_dir="raw/derived/preps",
        adapter=_fake_adapter,
    )

    assert result.succeeded
    assert result.prep_id == "paper-1"
    assert result.bundle_dir == "raw/derived/preps/paper-1"
    assert (
        result.manifest_path == "raw/derived/preps/paper-1/pdf_prep_manifest.yaml"
    )
    assert (ws.root / result.bundle_dir / "docling.md").is_file()


def test_prepare_pdf_derives_safe_prep_id_from_messy_filename(
    tmp_path: Path,
) -> None:
    ws = Workspace.init(tmp_path)
    pdf = ws.raw_sources / "papers" / "Smith et al. (2024).pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.7\n")

    result = prepare_pdf(ws, pdf_path=pdf, adapter=_fake_adapter)

    assert result.succeeded
    # Auto-derivation keeps `.`, `_`, `-` and collapses everything
    # else (whitespace, parens, etc.) to `-`; trailing / leading
    # separators are stripped. `.` is preserved because it is
    # filesystem-safe and useful for version-bearing names
    # (`paper.example`, `v1.2`). Users who want a cleaner id can pass
    # `--prep-id` explicitly.
    assert result.prep_id == "Smith-et-al.-2024"
    assert (ws.root / result.bundle_dir / "docling.md").is_file()


def test_prepare_pdf_refuses_output_dir_that_escapes_workspace(
    tmp_path: Path,
) -> None:
    """Cleanup-2026-05-14 regression: ``output_dir="../outside"`` must
    refuse with a structured `PdfPrepResult` and write nothing outside
    the workspace root. The CLI should never see an uncaught path
    traversal."""
    ws = Workspace.init(tmp_path)
    pdf = _write_pdf_under_workspace(ws)

    # Snapshot the parent of the workspace so we can confirm nothing
    # leaks past the workspace root.
    outside_parent = tmp_path.parent
    outside_before = sorted(p.name for p in outside_parent.iterdir())

    result = prepare_pdf(
        ws, pdf_path=pdf, output_dir="../outside", adapter=_fake_adapter
    )

    assert not result.succeeded
    assert result.status == "refused"
    assert result.refusal_reason is not None
    assert "workspace" in result.refusal_reason.lower()
    assert "../outside" in result.refusal_reason

    outside_after = sorted(p.name for p in outside_parent.iterdir())
    assert outside_before == outside_after, (
        f"refused output_dir wrote outside the workspace: "
        f"before={outside_before} after={outside_after}"
    )
    assert not (tmp_path.parent / "outside").exists()
    # And no bundle was created under the workspace either.
    assert not (ws.root / "raw" / "derived" / "pdf").exists() or list(
        (ws.root / "raw" / "derived" / "pdf").iterdir()
    ) == []


def test_prepare_pdf_refuses_absolute_output_dir(tmp_path: Path) -> None:
    """Cleanup-2026-05-14 regression: an absolute ``output_dir``
    (including drive-qualified paths on Windows) must refuse and
    write nothing outside the workspace."""
    ws = Workspace.init(tmp_path)
    pdf = _write_pdf_under_workspace(ws)

    elsewhere = tmp_path.parent / "elsewhere_for_prep"
    assert not elsewhere.exists()

    result = prepare_pdf(
        ws, pdf_path=pdf, output_dir=str(elsewhere), adapter=_fake_adapter
    )

    assert not result.succeeded
    assert result.status == "refused"
    assert result.refusal_reason is not None
    assert "workspace" in result.refusal_reason.lower()
    assert not elsewhere.exists(), "refused absolute output_dir created the path"


def test_prepare_pdf_missing_docling_retry_does_not_require_overwrite(
    tmp_path: Path,
) -> None:
    """Cleanup-2026-05-14 regression: a missing-Docling refusal must
    leave no sticky bundle directory. After the user installs the
    optional extra, retrying with the same `prep_id` succeeds without
    `--overwrite`."""
    ws = Workspace.init(tmp_path)
    pdf = _write_pdf_under_workspace(ws)

    def missing_adapter(_: Path) -> DoclingArtifacts:
        raise DoclingNotInstalledError(
            'docling is not installed. Install with: pip install "llloom[docling]"'
        )

    first = prepare_pdf(ws, pdf_path=pdf, adapter=missing_adapter)
    assert not first.succeeded
    assert first.status == "refused"
    assert first.refusal_reason is not None
    assert "llloom[docling]" in first.refusal_reason

    bundle = ws.root / "raw" / "derived" / "pdf" / "example"
    # No sticky bundle directory or contents from the missing-Docling refusal.
    assert not bundle.exists() or list(bundle.iterdir()) == []

    # Retry with the working adapter and the same prep id — no overwrite
    # flag should be required.
    second = prepare_pdf(ws, pdf_path=pdf, adapter=_fake_adapter)
    assert second.succeeded
    assert (bundle / "docling.md").read_text(encoding="utf-8") == _FAKE_MARKDOWN
    assert (bundle / "docling.json").is_file()
    assert (bundle / "pdf_prep_manifest.yaml").is_file()


def test_prepare_pdf_does_not_invoke_model_or_ingest(tmp_path: Path) -> None:
    """Sanity check: prepare_pdf must not exercise ingest/render/model paths.

    The op never imports the harness or any provider adapter and never
    touches `claims/`, `pages/`, or `state/source_registry/`. We assert
    the workspace is clean after a successful prep run except for the
    new `raw/derived/pdf/<prep_id>/` bundle plus the journal + lock the
    operation context manager creates.
    """
    ws = Workspace.init(tmp_path)
    pdf = _write_pdf_under_workspace(ws)

    result = prepare_pdf(ws, pdf_path=pdf, adapter=_fake_adapter)
    assert result.succeeded

    # No claim entity files, no rendered pages, no source-registry rows.
    assert list(ws.claims_entities.glob("*.yaml")) == []
    assert list((ws.pages / "concepts").glob("*.md")) == []
    assert list(ws.state_source_registry.glob("*.yaml")) == []
    # A journal entry for the op exists but the lock has been released.
    journals = list(ws.state_journals.glob("op.prepare_pdf.*.yaml"))
    assert len(journals) >= 1
    assert not (ws.state_locks / "workspace.yaml").exists()
