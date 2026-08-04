"""Contract tests: every model-bound LLMInvoke call leaves an
inspectable persisted audit trail.

Pre-hardening, ``ingest`` discarded the returned ``InvocationLog``.
This pass attaches the log to the operation journal entry and the
canary lint scan covers it because journals are already lint-scanned
verbatim for canary tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.ops.ingest import ingest
from llloom.ops.lint import FIXED_CANARY_TOKEN, lint
from llloom.state.journal import OperationJournal
from llloom.workspace.layout import Workspace


def _ingest_minimal(ws: Workspace) -> str:
    src = ws.raw_sources / "doc.md"
    src.write_text("# Doc\n\n## body\n\nSentence one.\n", encoding="utf-8")
    result = ingest(
        ws, src, source_id="src.doc", source_class="markdown_prose"
    )
    assert result.succeeded
    return result.op_id


def test_ingest_persists_invocation_log_into_journal(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    op_id = _ingest_minimal(ws)
    journal = OperationJournal(ws)
    entry = journal.load(op_id)
    assert entry.invocation_logs, (
        "expected at least one persisted invocation log on the journal entry"
    )
    log = entry.invocation_logs[0]
    assert log["operation_kind"] == "ingest"
    classes = sorted(item["class"] for item in log["read_inputs"])
    assert "SourceDocument" in classes
    assert "SchemaDocument" in classes
    assert log["model_identifier"]
    assert log["output_hash"].startswith("sha256:")


def test_invocation_log_records_only_hashes_not_raw_text(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    op_id = _ingest_minimal(ws)
    journal_yaml = (ws.state_journals / f"{op_id}.yaml").read_text(encoding="utf-8")
    # The raw source content must not appear inside the persisted journal.
    assert "Sentence one." not in journal_yaml, (
        "invocation log persisted raw source text; the audit trail must "
        "contain content hashes, not bodies"
    )


def test_canary_lint_scans_persisted_invocation_logs(tmp_path: Path) -> None:
    """Plant the fixed canary token inside the persisted ``invocation_logs``
    field on the journal entry and confirm lint flags it.

    The hardening review caveat called out the prior version of this
    test for planting the token in ``entry.notes``: that proved general
    journal scanning but did not exercise the specific
    ``invocation_logs`` path. Planting directly inside
    ``invocation_logs[0]`` proves the precise path requested.
    """
    ws = Workspace.init(tmp_path)
    op_id = _ingest_minimal(ws)
    journal = OperationJournal(ws)
    entry = journal.load(op_id)
    assert entry.invocation_logs, (
        "fixture precondition: ingest must persist at least one "
        "invocation log on the journal entry"
    )
    # Plant the canary directly inside the invocation_logs structure,
    # not into any other journal field.
    entry.invocation_logs[0]["poisoned_field"] = (
        f"contaminated: {FIXED_CANARY_TOKEN}"
    )
    journal.save(entry)

    # Sanity check: the token is now persisted inside invocation_logs
    # and is NOT elsewhere on the entry, so a passing assertion below
    # means lint scanned the invocation-log path specifically.
    persisted_yaml = (ws.state_journals / f"{op_id}.yaml").read_text(
        encoding="utf-8"
    )
    assert FIXED_CANARY_TOKEN in persisted_yaml
    assert FIXED_CANARY_TOKEN not in "\n".join(entry.notes)

    result = lint(ws)
    assert result.canary_hits, (
        "lint must flag canary tokens that appear inside the "
        "persisted invocation_logs field on a journal entry"
    )
    assert any("journals" in hit for hit in result.canary_hits)

