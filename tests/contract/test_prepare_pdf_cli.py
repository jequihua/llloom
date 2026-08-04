"""Contract tests for the ``llloom prepare-pdf`` CLI surface.

Pins three load-bearing properties of the new verb:

- the CLI subparser is registered and addressable through
  ``llloom.cli.main``;
- exercising the command end-to-end through ``main`` with a fake
  Docling adapter installed in `llloom.ops.prepare_pdf.convert_with_docling`
  writes the expected bundle and returns exit code 0;
- the CLI base path does not require the optional `llloom[docling]`
  extra: importing `llloom`, `llloom.cli`, and `llloom.pdf_prep` and
  building the argparse parser must work without `docling` installed.

The CLI guard also asserts that ``prepare-pdf`` is the only new verb
this slice added (the prior baseline had 16 verbs; the post-slice
total is 17).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from llloom.cli import _build_parser, main
from llloom.pdf_prep.docling import DoclingArtifacts
from llloom.workspace.layout import Workspace


_BASELINE_VERBS_BEFORE_4J_PDF = {
    "init",
    "status",
    "ingest",
    "verify",
    "render",
    "query",
    "lint",
    "reconcile",
    "unlock",
    "promote",
    "retract",
    "rebuild",
    "list_merge_proposals",
    "review-alias",
    "merge-alias",
    "reject-alias",
}
EXPECTED_VERBS = _BASELINE_VERBS_BEFORE_4J_PDF | {
    "prepare-pdf",
    "seed",
    # Slice 077 added five read-only listing / card verbs.
    "list-claims",
    "claim-card",
    "list-sources",
    "list-pages",
    "list-render-targets",
    # Slice 078 added one mutation verb.
    "supersede",
    # Slice 079 added one read-only diagnostic verb.
    "doctor",
    # Slice 084 added one mutation verb (page command group with
    # create subcommand).
    "page",
}


def test_cli_registers_prepare_pdf_verb_and_keeps_baseline() -> None:
    parser = _build_parser()
    sub = next(
        a for a in parser._actions if a.dest == "command"  # type: ignore[attr-defined]
    )
    registered = set(sub.choices.keys())  # type: ignore[attr-defined]
    assert "prepare-pdf" in registered
    assert "seed" in registered
    # Slice 077 verbs.
    assert "list-claims" in registered
    assert "claim-card" in registered
    assert "list-sources" in registered
    assert "list-pages" in registered
    assert "list-render-targets" in registered
    # Slice 078 verb.
    assert "supersede" in registered
    # Slice 079 verb.
    assert "doctor" in registered
    # Slice 084 verb (page command group with create subcommand).
    assert "page" in registered
    assert registered == EXPECTED_VERBS, (
        f"CLI verb set drifted; unexpected delta: "
        f"added={registered - EXPECTED_VERBS} "
        f"removed={EXPECTED_VERBS - registered}"
    )
    # Post-Slice-084 total: prior 25 + Slice 084's page = 26.
    assert len(registered) == 26


def test_cli_prepare_pdf_end_to_end_with_fake_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = Workspace.init(tmp_path)
    pdf = ws.raw_sources / "example.pdf"
    pdf.write_bytes(b"%PDF-1.7\n%fake\n")

    def fake_adapter(_: Path) -> DoclingArtifacts:
        return DoclingArtifacts(
            markdown="# CLI fake\n",
            json_text='{"name": "cli-fake"}',
            version="0.0.0-fake",
        )

    # Patch the default adapter resolution path so the CLI command,
    # which does not expose an `--adapter` flag, still avoids importing
    # docling. The op falls back to `convert_with_docling` when no
    # adapter is passed; monkeypatching that symbol on the op module
    # is the supported test seam.
    # `llloom.ops.__init__` re-exports the `prepare_pdf` function and
    # that shadows the `llloom.ops.prepare_pdf` submodule when accessed
    # via `getattr` (same shape as the `llloom.ops.lint` shadow
    # documented in `prompts/coder_handoff.md`). Resolve the module
    # explicitly so monkeypatch sets the attribute on the module.
    import importlib

    prepare_pdf_mod = importlib.import_module("llloom.ops.prepare_pdf")
    monkeypatch.setattr(prepare_pdf_mod, "convert_with_docling", fake_adapter)

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "prepare-pdf",
            str(pdf),
            "--prep-id",
            "cli-example",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["status"] == "succeeded"
    assert payload["prep_id"] == "cli-example"
    assert payload["selected_artifact"]["path"].endswith("docling.md")
    assert payload["op_id"].startswith("op.prepare_pdf.")

    bundle = ws.root / "raw" / "derived" / "pdf" / "cli-example"
    assert (bundle / "docling.md").read_text(encoding="utf-8") == "# CLI fake\n"
    assert (bundle / "docling.json").is_file()
    assert (bundle / "pdf_prep_manifest.yaml").is_file()

    # No claim or render artifacts produced: prep-pdf is not ingest.
    assert list(ws.claims_entities.glob("*.yaml")) == []
    assert list((ws.pages / "concepts").glob("*.md")) == []
    assert list(ws.state_source_registry.glob("*.yaml")) == []


def test_cli_prepare_pdf_refuses_existing_bundle_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace.init(tmp_path)
    pdf = ws.raw_sources / "example.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    def fake_adapter(_: Path) -> DoclingArtifacts:
        return DoclingArtifacts(
            markdown="# E\n", json_text="{}", version="x"
        )

    # `llloom.ops.__init__` re-exports the `prepare_pdf` function and
    # that shadows the `llloom.ops.prepare_pdf` submodule when accessed
    # via `getattr` (same shape as the `llloom.ops.lint` shadow
    # documented in `prompts/coder_handoff.md`). Resolve the module
    # explicitly so monkeypatch sets the attribute on the module.
    import importlib

    prepare_pdf_mod = importlib.import_module("llloom.ops.prepare_pdf")
    monkeypatch.setattr(prepare_pdf_mod, "convert_with_docling", fake_adapter)

    assert main(
        ["--root", str(tmp_path), "prepare-pdf", str(pdf), "--prep-id", "ex"]
    ) == 0
    # Second invocation refuses with non-zero exit.
    assert main(
        ["--root", str(tmp_path), "prepare-pdf", str(pdf), "--prep-id", "ex"]
    ) == 1


def test_cli_prepare_pdf_refuses_escaping_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cleanup-2026-05-14 regression: the CLI must refuse a path-traversal
    `--output-dir` with exit code 1 and a structured JSON `refusal_reason`."""
    ws = Workspace.init(tmp_path)
    pdf = ws.raw_sources / "example.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    def fake_adapter(_: Path) -> DoclingArtifacts:
        return DoclingArtifacts(
            markdown="# E\n", json_text="{}", version="x"
        )

    import importlib

    prepare_pdf_mod = importlib.import_module("llloom.ops.prepare_pdf")
    monkeypatch.setattr(prepare_pdf_mod, "convert_with_docling", fake_adapter)

    outside_parent = tmp_path.parent
    outside_before = sorted(p.name for p in outside_parent.iterdir())

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "prepare-pdf",
            str(pdf),
            "--prep-id",
            "ex",
            "--output-dir",
            "../outside",
        ]
    )
    assert exit_code == 1
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["status"] == "refused"
    assert payload["refusal_reason"]
    assert "workspace" in payload["refusal_reason"].lower()
    assert "../outside" in payload["refusal_reason"]

    outside_after = sorted(p.name for p in outside_parent.iterdir())
    assert outside_before == outside_after, (
        f"CLI refusal wrote outside the workspace: "
        f"before={outside_before} after={outside_after}"
    )
    assert not (tmp_path.parent / "outside").exists()


def test_base_import_does_not_require_docling() -> None:
    """Regression: importing `llloom`, `llloom.cli`, and
    `llloom.pdf_prep` must work without the optional `docling` package.
    The CLI parser must build without triggering the lazy import inside
    ``llloom.pdf_prep.docling.convert_with_docling``.
    """
    if "docling" in sys.modules:
        # Some environments pre-import docling for unrelated reasons.
        # The contract is about *not requiring* docling; if it happens
        # to be installed we can still assert the import surface.
        pass
    else:
        # Defensive: ensure no submodule grabbed `docling` eagerly.
        assert all(not m.startswith("docling") for m in sys.modules)

    # Importing `llloom.pdf_prep` should not import `docling`.
    importlib.import_module("llloom")
    pdf_prep = importlib.import_module("llloom.pdf_prep")
    cli = importlib.import_module("llloom.cli")
    parser = cli._build_parser()
    assert parser is not None
    assert hasattr(pdf_prep, "convert_with_docling")

    # The op module imports `convert_with_docling` from
    # `llloom.pdf_prep.docling`; that path must also not eagerly import
    # the `docling` SDK.
    prepare_pdf_mod = importlib.import_module("llloom.ops.prepare_pdf")
    assert hasattr(prepare_pdf_mod, "prepare_pdf")
