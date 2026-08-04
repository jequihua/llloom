"""`prepare-pdf` operation.

Optional first-party PDF working-text prep. Produces a deterministic
bundle under ``raw/derived/pdf/<prep_id>/`` containing the selected
ingest artifact (`docling.md`), its structured-export sibling
(`docling.json`), and a provider-neutral manifest
(`pdf_prep_manifest.yaml`). The selected artifact is registered later
through the existing ``llloom ingest`` command — this op never
ingests, never renders, never invokes a model.

The Docling adapter is lazy-imported inside
``llloom.pdf_prep.convert_with_docling``; tests inject fake adapters
through the `adapter` parameter so the default suite passes without
the optional `llloom[docling]` extra installed.

See `02_analysis/docling_default_pdf_prep_milestone.md` for the
milestone contract and the provider-neutral manifest shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from llloom.ops._context import operation, relative_posix
from llloom.ops.results import PdfPrepArtifact, PdfPrepResult
from llloom.pdf_prep.docling import (
    DoclingArtifacts,
    DoclingConversionError,
    DoclingNotInstalledError,
    convert_with_docling,
)
from llloom.pdf_prep.manifest import (
    MANIFEST_FILENAME,
    PROVIDER_DOCLING_DEFAULT,
    SELECTED_ARTIFACT_KIND,
    build_manifest,
    sha256_of_file,
    write_manifest,
)
from llloom.workspace.layout import Workspace

DEFAULT_OUTPUT_DIR = "raw/derived/pdf"
SELECTED_MARKDOWN_FILENAME = "docling.md"
SELECTED_JSON_FILENAME = "docling.json"

_PREP_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


class PrepIdError(ValueError):
    """Raised when a supplied or derived prep id is not filesystem-safe."""


class OutputDirError(ValueError):
    """Raised when ``output_dir`` is not workspace-relative.

    Caught inside :func:`prepare_pdf` and surfaced as a structured
    ``PdfPrepResult`` with ``status="refused"`` so the CLI never raises
    on bad user input. The refusal reason names the offending
    ``output_dir`` and the workspace-containment rule.
    """


def _validate_output_dir(workspace_root: Path, output_dir_rel: str) -> None:
    """Refuse any ``output_dir`` that is not workspace-relative.

    Rejects absolute paths (POSIX or Windows), drive-qualified
    paths (`C:foo`), UNC paths (`\\\\server\\share\\...`), paths
    rooted at `/` or `\\` (POSIX-style or backslash on Windows),
    and otherwise-relative paths whose resolved form sits outside
    ``workspace_root``. ``..`` is permitted as long as the final
    resolved path stays inside the workspace.
    """
    candidate = Path(output_dir_rel)
    if candidate.is_absolute() or candidate.drive or candidate.root:
        raise OutputDirError(
            f"output_dir {output_dir_rel!r} must be workspace-relative; "
            "absolute, drive-qualified, UNC, and root-prefixed paths "
            "are refused so output directories stay inside the workspace"
        )
    workspace_resolved = workspace_root.resolve()
    candidate_resolved = (workspace_root / candidate).resolve()
    try:
        candidate_resolved.relative_to(workspace_resolved)
    except ValueError as exc:
        raise OutputDirError(
            f"output_dir {output_dir_rel!r} resolves outside the "
            f"workspace root ({workspace_resolved}); output directories "
            "must stay inside the workspace"
        ) from exc


def _derive_prep_id(pdf_path: Path) -> str:
    """Derive a filesystem-safe prep id from the PDF filename stem.

    Collapses every unsafe character to ``-`` and strips leading
    / trailing separators. Refuses empty results.
    """
    stem = pdf_path.stem
    safe = _PREP_ID_RE.sub("-", stem).strip("-._")
    if not safe:
        raise PrepIdError(
            f"cannot derive prep id from PDF filename {pdf_path.name!r}; "
            "pass --prep-id explicitly"
        )
    return safe


def _validate_prep_id(prep_id: str) -> None:
    if not prep_id:
        raise PrepIdError("prep id must not be empty")
    if _PREP_ID_RE.search(prep_id) or prep_id.startswith((".", "-", "_")):
        raise PrepIdError(
            f"prep id {prep_id!r} contains characters outside [A-Za-z0-9._-] "
            "or starts with a separator"
        )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _refused(
    *,
    prep_id: str,
    bundle_dir_rel: str,
    manifest_path_rel: str,
    source_pdf_rel: str,
    source_pdf_sha256: str,
    reason: str,
    op_id: str,
) -> PdfPrepResult:
    return PdfPrepResult(
        prep_id=prep_id,
        status="refused",
        bundle_dir=bundle_dir_rel,
        manifest_path=manifest_path_rel,
        source_pdf=source_pdf_rel,
        source_pdf_sha256=source_pdf_sha256,
        provider=PROVIDER_DOCLING_DEFAULT,
        refusal_reason=reason,
        op_id=op_id,
    )


def prepare_pdf(
    workspace: Workspace,
    *,
    pdf_path: Path | str,
    prep_id: str | None = None,
    output_dir: str | None = None,
    overwrite: bool = False,
    adapter: Callable[[Path], DoclingArtifacts] | None = None,
) -> PdfPrepResult:
    """Prepare a PDF into a deterministic artifact bundle.

    The bundle is written under
    ``<workspace>/<output_dir>/<prep_id>/`` and contains
    ``docling.md``, ``docling.json``, and
    ``pdf_prep_manifest.yaml``. The manifest selects ``docling.md`` as
    the ingest artifact for a later normal ``llloom ingest`` call.

    Failure modes:

    - ``output_dir`` is absolute, drive-qualified, UNC, or resolves
      outside the workspace root: refusal naming the rule.
    - Missing optional dependency: returns a `PdfPrepResult` with
      `status="refused"` and a reason naming ``llloom[docling]``.
      The bundle directory is **not** created in this case, so a
      retry after `pip install "llloom[docling]"` succeeds without
      `--overwrite`.
    - Pre-existing bundle without ``overwrite=True``: refusal.
    - Missing or unreadable PDF: refusal.
    - Docling raises during conversion: a failed-status manifest is
      written (so the journal audit trail is honest); a later
      retry requires ``overwrite=True``.
    """
    pdf_path = Path(pdf_path)
    output_dir_rel = (output_dir or DEFAULT_OUTPUT_DIR).strip()
    if not output_dir_rel:
        raise ValueError("output_dir must not be empty")

    resolved_prep_id = prep_id or _derive_prep_id(pdf_path)
    _validate_prep_id(resolved_prep_id)

    bundle_dir = workspace.root / output_dir_rel / resolved_prep_id
    bundle_dir_rel = f"{output_dir_rel}/{resolved_prep_id}"
    manifest_path = bundle_dir / MANIFEST_FILENAME
    manifest_path_rel = f"{bundle_dir_rel}/{MANIFEST_FILENAME}"
    markdown_path = bundle_dir / SELECTED_MARKDOWN_FILENAME
    json_path = bundle_dir / SELECTED_JSON_FILENAME

    source_pdf_rel = relative_posix(workspace, pdf_path)

    with operation(workspace, op_kind="prepare_pdf") as ctx:
        try:
            _validate_output_dir(workspace.root, output_dir_rel)
        except OutputDirError as exc:
            return _refused(
                prep_id=resolved_prep_id,
                bundle_dir_rel=bundle_dir_rel,
                manifest_path_rel=manifest_path_rel,
                source_pdf_rel=source_pdf_rel,
                source_pdf_sha256="",
                reason=str(exc),
                op_id=ctx.op_id,
            )

        if not pdf_path.is_file():
            return _refused(
                prep_id=resolved_prep_id,
                bundle_dir_rel=bundle_dir_rel,
                manifest_path_rel=manifest_path_rel,
                source_pdf_rel=source_pdf_rel,
                source_pdf_sha256="",
                reason=f"source PDF not found: {pdf_path}",
                op_id=ctx.op_id,
            )

        source_sha = sha256_of_file(pdf_path)

        if bundle_dir.exists() and not overwrite:
            return _refused(
                prep_id=resolved_prep_id,
                bundle_dir_rel=bundle_dir_rel,
                manifest_path_rel=manifest_path_rel,
                source_pdf_rel=source_pdf_rel,
                source_pdf_sha256=source_sha,
                reason=(
                    f"prep bundle already exists at {bundle_dir_rel}; "
                    "pass --overwrite to replace"
                ),
                op_id=ctx.op_id,
            )

        # Bundle directory creation is deferred until after the adapter
        # actually has something to write (success path) or until a
        # failed-status manifest must be persisted for audit (handled
        # inside `write_manifest`). On `DoclingNotInstalledError` no
        # directory is created, so a later retry after the user
        # installs `llloom[docling]` succeeds without `--overwrite`.

        run_adapter = adapter if adapter is not None else convert_with_docling
        try:
            artifacts = run_adapter(pdf_path)
        except DoclingNotInstalledError as exc:
            return _refused(
                prep_id=resolved_prep_id,
                bundle_dir_rel=bundle_dir_rel,
                manifest_path_rel=manifest_path_rel,
                source_pdf_rel=source_pdf_rel,
                source_pdf_sha256=source_sha,
                reason=str(exc),
                op_id=ctx.op_id,
            )
        except DoclingConversionError as exc:
            failed_manifest = build_manifest(
                prep_id=resolved_prep_id,
                status="failed",
                source_pdf_workspace_path=source_pdf_rel,
                source_pdf_sha256=source_sha,
                artifacts=[],
                selected_ingest_artifact=None,
                docling_status="failed",
                docling_version="unknown",
            )
            write_manifest(manifest_path, failed_manifest)
            ctx.entry.touched_files.append(manifest_path_rel)
            return PdfPrepResult(
                prep_id=resolved_prep_id,
                status="failed",
                bundle_dir=bundle_dir_rel,
                manifest_path=manifest_path_rel,
                source_pdf=source_pdf_rel,
                source_pdf_sha256=source_sha,
                provider=PROVIDER_DOCLING_DEFAULT,
                components={"docling": "failed"},
                refusal_reason=str(exc),
                op_id=ctx.op_id,
            )

        _atomic_write_text(markdown_path, artifacts.markdown)
        _atomic_write_text(json_path, artifacts.json_text)

        markdown_sha = sha256_of_file(markdown_path)
        json_sha = sha256_of_file(json_path)

        markdown_rel = f"{bundle_dir_rel}/{SELECTED_MARKDOWN_FILENAME}"
        json_rel = f"{bundle_dir_rel}/{SELECTED_JSON_FILENAME}"

        artifact_records: list[dict] = [
            {"path": markdown_rel, "kind": SELECTED_ARTIFACT_KIND, "sha256": markdown_sha},
            {"path": json_rel, "kind": "docling_json", "sha256": json_sha},
        ]
        selected_record: dict = {
            "path": markdown_rel,
            "kind": SELECTED_ARTIFACT_KIND,
            "sha256": markdown_sha,
        }

        manifest = build_manifest(
            prep_id=resolved_prep_id,
            status="succeeded",
            source_pdf_workspace_path=source_pdf_rel,
            source_pdf_sha256=source_sha,
            artifacts=artifact_records,
            selected_ingest_artifact=selected_record,
            docling_status="succeeded",
            docling_version=artifacts.version,
        )
        write_manifest(manifest_path, manifest)

        for rel in (markdown_rel, json_rel, manifest_path_rel):
            ctx.entry.touched_files.append(rel)

        result = PdfPrepResult(
            prep_id=resolved_prep_id,
            status="succeeded",
            bundle_dir=bundle_dir_rel,
            manifest_path=manifest_path_rel,
            source_pdf=source_pdf_rel,
            source_pdf_sha256=source_sha,
            provider=PROVIDER_DOCLING_DEFAULT,
            artifacts=[
                PdfPrepArtifact(**a) for a in artifact_records
            ],
            components={"docling": "succeeded"},
            selected_artifact=PdfPrepArtifact(**selected_record),
            op_id=ctx.op_id,
        )
        return result
