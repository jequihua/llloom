"""Contract: Graphab operator-friction cleanup (Slice 081).

Pins the three CLI behaviors the Graphab review-agent feedback
asked for (see
``feedback/2026-05-24_graphab_llloom_feedback_next_priorities.md``):

1. ``llloom --root <ws> ingest <relative>`` resolves the
   relative source path against the workspace root, not the
   shell cwd. Absolute paths stay absolute. The existing
   inside-the-memory-root flow stays byte-compatible because
   ``cwd == workspace.root`` makes the two resolutions agree.
2. A missing source path produces a clean stderr diagnostic
   naming both the resolved path and the workspace root, and
   exits 1 with no Python traceback.
3. Expected render / page-marker failures
   (``PageParseError``, ``RenderError``) surface as concise
   stderr messages and exit 1 with no traceback.

No new CLI verb, no new dataclass, no library exception
behavior change.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llloom.claims.models import Locator
from llloom.cli import main as cli_main
from llloom.ops.ingest import SeedClaim, ingest
from llloom.sources.registry import SourceRegistry
from llloom.workspace.layout import Workspace


SOURCE_TEXT = """\
# Article

## Methods

Alpha is documented in the source. It anchors the Slice 081 cleanup.
"""


GOOD_PAGE = """\
---
page_id: concept/alpha
page_class: concept
write_policy: mixed
status: rendered
---

<!-- llloom:claim-block id=claim_block.concept.alpha -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.alpha owner=human -->
Commentary alpha.
<!-- /llloom:commentary -->
"""


MALFORMED_PAGE = """\
---
page_id: concept/alpha
page_class: concept
write_policy: mixed
status: rendered
---

<!-- llloom:claim-block id=claim_block.concept.alpha -->
placeholder
<!-- /llloom:claim-block -->
"""


def _seed_workspace(tmp_path: Path) -> Path:
    """Initialize a memory workspace at ``tmp_path / 'memory'``
    with one Markdown source under ``raw/sources/`` and one
    valid variant-(B) page under ``pages/concepts/alpha.md``.
    Returns the memory root path.
    """
    memory = tmp_path / "memory"
    ws = Workspace.init(memory)
    src = ws.raw_sources / "source.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    page = ws.pages / "concepts" / "alpha.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(GOOD_PAGE, encoding="utf-8")
    return memory


def _seed_claim_for_alpha(memory: Path) -> None:
    """Seed one claim against ``concept/alpha`` so a render call
    has something to write into the claim block.
    """
    ws = Workspace.load(memory)
    seed = SeedClaim(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        claim_id="c.alpha.1",
        claim_kind="definition",
        claim_text="Alpha is documented in the source. It anchors the Slice 081 cleanup.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=2,
        ),
        render_target=("concept/alpha", "claim_block.concept.alpha"),
    )
    ingest(
        ws,
        ws.raw_sources / "source.md",
        source_id="src.source",
        source_class="markdown_prose",
        seed_claims=[seed],
    )


# ---------------------------------------------------------------------
# 1. Root-relative CLI ingest from outside the memory root
# ---------------------------------------------------------------------


def test_cli_ingest_resolves_relative_path_against_root_from_outside(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    memory = _seed_workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    rc = cli_main(
        [
            "--root",
            str(memory),
            "ingest",
            "raw/sources/source.md",
            "--source-id",
            "src.source",
            "--source-class",
            "markdown_prose",
            "--no-render",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    ws = Workspace.load(memory)
    registry = SourceRegistry(ws)
    record = registry.load("src.source")
    assert record.raw_path.endswith("raw/sources/source.md")


# ---------------------------------------------------------------------
# 2. Inside-the-memory-root flow stays byte-compatible
# ---------------------------------------------------------------------


def test_cli_ingest_inside_root_remains_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    memory = _seed_workspace(tmp_path)
    monkeypatch.chdir(memory)

    rc = cli_main(
        [
            "--root",
            ".",
            "ingest",
            "raw/sources/source.md",
            "--source-id",
            "src.source",
            "--source-class",
            "markdown_prose",
            "--no-render",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err


# ---------------------------------------------------------------------
# 3. Missing relative path exits cleanly with no traceback
# ---------------------------------------------------------------------


def test_cli_ingest_missing_relative_path_exits_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    memory = _seed_workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    rc = cli_main(
        [
            "--root",
            str(memory),
            "ingest",
            "raw/sources/does_not_exist.md",
            "--source-id",
            "src.missing",
            "--source-class",
            "markdown_prose",
            "--no-render",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
    assert "source path not found" in captured.err
    # The diagnostic must name both the resolved path AND the
    # workspace root so the operator can immediately see whether
    # --root was correctly set.
    assert "raw/sources/does_not_exist.md" in captured.err.replace("\\", "/")
    # ``ws.root.resolve()`` (used in the diagnostic) on Windows
    # uses backslashes; normalize before substring-asserting.
    assert "memory" in captured.err.replace("\\", "/")


# ---------------------------------------------------------------------
# 4. Render marker failure exits cleanly with no traceback
# ---------------------------------------------------------------------


def test_cli_render_marker_failure_exits_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    memory = _seed_workspace(tmp_path)
    _seed_claim_for_alpha(memory)

    # Replace the previously-valid page with a malformed one so
    # ``render page:concept/alpha`` parses the broken markers and
    # ``RenderError`` propagates. (The malformed page omits the
    # commentary marker pair entirely.)
    (memory / "pages" / "concepts" / "alpha.md").write_text(
        MALFORMED_PAGE, encoding="utf-8"
    )

    rc = cli_main(
        ["--root", str(memory), "render", "page:concept/alpha"]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
    assert "llloom render:" in captured.err
    assert "next:" in captured.err
    assert "render --dry-run" in captured.err


# ---------------------------------------------------------------------
# 5. Render --dry-run on the same malformed page also exits cleanly
# ---------------------------------------------------------------------


def test_cli_render_dry_run_marker_failure_exits_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dry-run path runs ``_build_inspection_result`` which
    catches ``PageParseError`` internally and surfaces it via
    ``marker_health="parse_error"``. Slice 081 leaves that
    behavior intact — the mutating render path is what raises
    ``RenderError`` and what the new CLI catch-block exists to
    handle. This test pins that dry-run still exits 0 on a
    malformed page (the diagnostic surfaces inside the plan, not
    via stderr).
    """
    memory = _seed_workspace(tmp_path)
    _seed_claim_for_alpha(memory)
    (memory / "pages" / "concepts" / "alpha.md").write_text(
        MALFORMED_PAGE, encoding="utf-8"
    )

    rc = cli_main(
        [
            "--root",
            str(memory),
            "render",
            "--dry-run",
            "page:concept/alpha",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    # JSON output should report parse_error marker health.
    assert "parse_error" in captured.out
