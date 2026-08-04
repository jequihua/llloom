"""Contract: render target validation runs lockless.

Pins the Slice 068 fix from
``feedback/2026-05-22_llloom_development_roadmap_synthesis.md``:
unknown or missing render targets must raise before
``WorkspaceLock.acquire`` is called, before any journal entry is
opened, and before any page or fingerprint write. Successful render
must still run under the workspace lock and preserve the variant-(B)
commentary region byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.cli import main as cli_main
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.render import render
from llloom.pages.regions import parse_page
from llloom.state.lock import WorkspaceLock
from llloom.workspace.layout import Workspace


PAGE_TEMPLATE = """\
---
page_id: concept/preflight
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.concept.preflight -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.preflight owner=human -->

Commentary that must survive byte-for-byte across rerender.

<!-- /llloom:commentary -->
"""

SOURCE_TEXT = """\
# Article

## Methods

Preflight validates render targets without acquiring the workspace lock. It refuses bad targets cheaply.

Second paragraph irrelevant.
"""


def _seed_real_claim(ws: Workspace) -> None:
    src = ws.raw_sources / "preflight.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    page_path = ws.pages / "concepts" / "preflight.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    seed = SeedClaim(
        entity_id="concept.preflight",
        entity_type="concept",
        display_name="Preflight",
        claim_id="c_0001",
        claim_kind="definition",
        claim_text=(
            "Preflight validates render targets without acquiring the "
            "workspace lock."
        ),
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("concept/preflight", "claim_block.concept.preflight"),
    )
    result = ingest(
        ws,
        src,
        source_id="src.preflight",
        source_class="markdown_prose",
        seed_claims=[seed],
    )
    assert result.succeeded, result.refusal_reason


def _no_lock_file(ws: Workspace) -> bool:
    return not (ws.state_locks / "workspace.yaml").is_file()


def _journal_files(ws: Workspace) -> list[Path]:
    if not ws.state_journals.is_dir():
        return []
    return sorted(ws.state_journals.glob("*.yaml"))


def test_unknown_target_does_not_acquire_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace.init(tmp_path)

    def _refuse(*args, **kwargs):
        raise AssertionError(
            "WorkspaceLock.acquire was called during preflight refusal"
        )

    monkeypatch.setattr(WorkspaceLock, "acquire", _refuse)

    with pytest.raises(ValueError) as excinfo:
        render(ws, target="concept/missing")

    msg = str(excinfo.value)
    assert "concept/missing" in msg
    assert "AssertionError" not in msg


def test_unknown_target_creates_no_journal_entry(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    before = _journal_files(ws)
    with pytest.raises(ValueError):
        render(ws, target="concept/missing")
    after = _journal_files(ws)
    assert before == after
    assert not any("render" in p.stem for p in after)


def test_unknown_target_leaves_no_lock_file(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    with pytest.raises(ValueError):
        render(ws, target="concept/missing")
    assert _no_lock_file(ws)


def test_missing_page_target_does_not_acquire_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace.init(tmp_path)

    def _refuse(*args, **kwargs):
        raise AssertionError(
            "WorkspaceLock.acquire was called for a missing page target"
        )

    monkeypatch.setattr(WorkspaceLock, "acquire", _refuse)

    with pytest.raises(ValueError) as excinfo:
        render(ws, target="page:does/not/exist")

    msg = str(excinfo.value)
    assert "page:does/not/exist" in msg
    assert _no_lock_file(ws)
    assert not _journal_files(ws)


def test_invalid_target_error_suggests_page_prefix(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        render(ws, target="concept/foo")
    msg = str(excinfo.value)
    assert "page:concept/foo" in msg
    assert "page:<page_id>" in msg


def test_valid_page_target_still_renders_under_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace.init(tmp_path)
    _seed_real_claim(ws)

    calls: list[str] = []
    real_acquire = WorkspaceLock.acquire

    def _spy(self, *args, **kwargs):
        calls.append(kwargs.get("op_id", "<unknown>"))
        return real_acquire(self, *args, **kwargs)

    monkeypatch.setattr(WorkspaceLock, "acquire", _spy)

    result = render(ws, target="page:concept/preflight")

    touched = result.rendered_pages + result.unchanged_pages
    assert any(p.endswith("preflight.md") for p in touched), touched
    assert calls, "valid render must acquire the workspace lock"
    assert any(op_id.startswith("op.render.") for op_id in calls)

    page_text = (ws.pages / "concepts" / "preflight.md").read_text(encoding="utf-8")
    parsed = parse_page(page_text)
    assert "Commentary that must survive byte-for-byte" in parsed.commentary_inner
    assert "claim:c_0001" in parsed.claim_block_inner


def test_cli_unknown_target_returns_nonzero_with_clean_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Workspace.init(tmp_path)
    rc = cli_main(["--root", str(tmp_path), "render", "concept/foo"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "Traceback" not in captured.err
    assert "page:concept/foo" in captured.err
    assert _no_lock_file(Workspace.load(tmp_path))
