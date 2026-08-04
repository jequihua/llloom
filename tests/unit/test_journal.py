"""Unit tests for `OperationJournal.new_op_id` uniqueness.

Pre-cleanup, the op id was ``op.<kind>.<YYYYMMDDTHHMMSSZ>`` at
second resolution. Two rapid same-kind promote calls observed in
Milestone 4I produced the same id; because journal entries live at
``state/journals/<op_id>.yaml``, a same-second collision could
overwrite an earlier journal file and leave fewer journal records
than operations performed.

The first cleanup added an in-process counter; the architect review
caught that the counter resets to ``.001`` in every new Python
process, so rapid separate ``llloom.exe`` CLI calls could still
collide on the same wall-clock second. This file pins the
cross-process contract by adding microsecond timestamp + PID + counter
to the id shape.

These tests pin the current contract:

- 100 rapid same-kind in-process calls return 100 unique ids;
- 10 rapid same-kind ids generated from separate Python subprocesses
  are also unique (regression for the architect's reproduction);
- every id keeps the ``op.<kind>.`` prefix so prefix checks (e.g.
  ``op.rebuild.health_report.``) still match;
- ids are filesystem-safe and round-trip through ``journal.path``;
- two rapid in-process ``journal.start`` calls do not overwrite each
  other's journal file;
- manually supplied bare-second ids
  (``op.ingest.20990101T000000Z``) still save and load correctly,
  preserving compatibility with externally specified ids.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from llloom.state.journal import JournalEntry, OperationJournal
from llloom.workspace.layout import Workspace


def test_new_op_id_uniqueness_in_rapid_loop() -> None:
    ids = [OperationJournal.new_op_id("promote") for _ in range(100)]
    assert len(ids) == 100
    assert len(set(ids)) == 100, "rapid same-kind new_op_id calls must be unique"


def test_new_op_id_preserves_kind_prefix() -> None:
    for _ in range(50):
        oid = OperationJournal.new_op_id("rebuild.health_report")
        assert oid.startswith("op.rebuild.health_report."), oid


def test_new_op_id_filesystem_safe(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    journal = OperationJournal(ws)
    for _ in range(20):
        oid = OperationJournal.new_op_id("promote")
        p = journal.path(oid)
        # parent must be state_journals, suffix .yaml, stem == oid
        assert p.parent == ws.state_journals
        assert p.suffix == ".yaml"
        assert p.stem == oid
        # touch the file via the supported save path to confirm
        # the os accepts the name on Windows / posix
        entry = JournalEntry(op_id=oid, op_kind="promote", status="in_progress")
        journal.save(entry)
        assert p.is_file()


def test_rapid_journal_starts_do_not_overwrite(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    journal = OperationJournal(ws)
    # Generate two ids back-to-back (almost guaranteed same second).
    a = OperationJournal.new_op_id("promote")
    b = OperationJournal.new_op_id("promote")
    assert a != b
    journal.start(op_id=a, op_kind="promote", lock_id="lock.workspace")
    journal.start(op_id=b, op_kind="promote", lock_id="lock.workspace")
    assert journal.exists(a)
    assert journal.exists(b)
    loaded_a = journal.load(a)
    loaded_b = journal.load(b)
    assert loaded_a.op_id == a
    assert loaded_b.op_id == b


def test_manually_supplied_bare_second_op_id_round_trips(tmp_path: Path) -> None:
    """Externally supplied op ids without a counter suffix
    (e.g. the historical ``op.ingest.20990101T000000Z`` shape used as
    test fixtures elsewhere in the suite) must still save and load
    through the journal unchanged."""
    ws = Workspace.init(tmp_path)
    journal = OperationJournal(ws)
    manual = "op.ingest.20990101T000000Z"
    entry = journal.start(op_id=manual, op_kind="ingest", lock_id="lock.workspace")
    journal.complete(entry, touched_files=[])
    loaded = journal.load(manual)
    assert loaded.op_id == manual
    assert loaded.status == "completed"


def test_new_op_id_unique_across_separate_processes(tmp_path: Path) -> None:
    """Regression for the architect-observed cross-process collision.

    The previous in-process counter reset to ``.001`` in every fresh
    Python interpreter, so rapid CLI invocations within the same
    wall-clock second produced colliding ids
    (``op.promote.20260514T114308Z.001`` repeated five times). The
    current shape includes microsecond resolution and the OS PID, so
    separate Python processes always produce distinct ids even when
    invoked in a tight loop.
    """
    program = (
        "from llloom.state.journal import OperationJournal; "
        "print(OperationJournal.new_op_id('promote'))"
    )
    ids: list[str] = []
    for _ in range(10):
        out = subprocess.check_output(
            [sys.executable, "-c", program],
            text=True,
            timeout=30,
        ).strip()
        ids.append(out)

    assert len(ids) == 10
    assert len(set(ids)) == 10, f"cross-process op ids must be unique: {ids}"
    for oid in ids:
        assert oid.startswith("op.promote."), oid
        # Filesystem-safe: id is used as the stem of
        # state/journals/<oid>.yaml in real operations.
        # On Windows the reserved characters are <>:"/\|?*; none are
        # produced by the new shape.
        assert all(ch not in oid for ch in '<>:"/\\|?*'), oid
        # Each id must be acceptable as a path stem under tmp_path.
        (tmp_path / f"{oid}.yaml").write_text("ok", encoding="utf-8")
