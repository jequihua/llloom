"""Contract: render plan, dry run, and target discovery (Slice 073).

Pins the agent-friendly inspection surface that lets a caller ask
"what would render do?" without acquiring the workspace lock,
opening a render journal entry, or writing any page or fingerprint:

- `render(..., dry_run=True)` and CLI `llloom render --dry-run`
  populate `RenderResult.plan` with one `RenderPlanEntry` per page,
  including contributors, claim ids, marker health, and
  would-change flags; no workspace mutation.
- `render(..., list_targets=True)` and CLI `llloom render
  --list-targets` provide the same plan output and additionally
  surface page-on-disk-with-no-contributors as a valid target with
  empty contributors and `marker_health="ok"`.
- target preflight raises the same `ValueError` on the read-only
  paths as on the mutating path (Slice 068 contract preserved).
- normal `render(...)` still acquires the lock and writes pages /
  fingerprints exactly as before (Slices 068 / 071 / 072 contracts
  preserved).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.cli import main as cli_main
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.render import render
from llloom.ops.results import RenderPlanContributor, RenderPlanEntry
from llloom.state.fingerprints import FingerprintStore
from llloom.state.lock import WorkspaceLock
from llloom.workspace.layout import Workspace


PAGE_TEMPLATE = """\
---
page_id: concept/shared
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.concept.shared -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.shared owner=human -->

Commentary that must survive byte-for-byte across rerender.

<!-- /llloom:commentary -->
"""


PAGE_TEMPLATE_LONELY = """\
---
page_id: concept/lonely
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.concept.lonely -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.lonely owner=human -->

A page on disk with no claim contributors. List-targets must still
surface it.

<!-- /llloom:commentary -->
"""


PAGE_TEMPLATE_MALFORMED = """\
---
page_id: concept/malformed
page_class: concept
write_policy: mixed
status: draft
---

No markers at all — this page should report parse_error on the
dry-run/list-targets read-only path and fail hard on real render.
"""


SOURCE_ALPHA = """\
# Article

## Methods

Alpha entity contributes the first verifiable claim about the shared block.

Second paragraph irrelevant.
"""


SOURCE_BETA = """\
# Article

## Methods

Beta entity contributes the second verifiable claim about the shared block.

Second paragraph irrelevant.
"""


def _seed_two_entity_workspace(tmp_path: Path) -> Workspace:
    """Two entities contributing to the same shared page/block."""
    ws = Workspace.init(tmp_path)
    src_a = ws.raw_sources / "alpha.md"
    src_a.write_text(SOURCE_ALPHA, encoding="utf-8")
    src_b = ws.raw_sources / "beta.md"
    src_b.write_text(SOURCE_BETA, encoding="utf-8")
    page_path = ws.pages / "concepts" / "shared.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    locator = Locator(
        locator_type="markdown_prose_v1",
        heading_path=["Methods"],
        paragraph_index=1,
        sentence_start=1,
        sentence_end=1,
    )
    ingest(
        ws,
        src_a,
        source_id="src.alpha",
        source_class="markdown_prose",
        seed_claims=[
            SeedClaim(
                entity_id="concept.alpha",
                entity_type="concept",
                display_name="Alpha",
                claim_id="c.alpha.1",
                claim_kind="definition",
                claim_text=(
                    "Alpha entity contributes the first verifiable claim "
                    "about the shared block."
                ),
                locator=locator,
                render_target=(
                    "concept/shared",
                    "claim_block.concept.shared",
                ),
            )
        ],
    )
    ingest(
        ws,
        src_b,
        source_id="src.beta",
        source_class="markdown_prose",
        seed_claims=[
            SeedClaim(
                entity_id="concept.beta",
                entity_type="concept",
                display_name="Beta",
                claim_id="c.beta.1",
                claim_kind="definition",
                claim_text=(
                    "Beta entity contributes the second verifiable claim "
                    "about the shared block."
                ),
                locator=locator,
                render_target=(
                    "concept/shared",
                    "claim_block.concept.shared",
                ),
            )
        ],
    )
    return ws


def _journal_files(ws: Workspace) -> list[str]:
    if not ws.state_journals.is_dir():
        return []
    return sorted(p.name for p in ws.state_journals.glob("*.yaml"))


def _capture_workspace_state(ws: Workspace) -> dict:
    page_path = ws.pages / "concepts" / "shared.md"
    fp_path = ws.render_fingerprints
    lock_path = ws.state_locks / "workspace.yaml"
    return {
        "page_text": page_path.read_text(encoding="utf-8"),
        "fingerprints_exists": fp_path.is_file(),
        "fingerprints_text": (
            fp_path.read_text(encoding="utf-8") if fp_path.is_file() else None
        ),
        "lock_exists": lock_path.is_file(),
        "journal_files": _journal_files(ws),
    }


def test_dry_run_writes_nothing_but_reports_plan(tmp_path: Path) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    # Corrupt the stored fingerprint so the dry-run sees a real
    # would-change flag for the shared page.
    FingerprintStore(ws).set("concept/shared", "sha256:deadbeef")
    before = _capture_workspace_state(ws)

    result = render(ws, dry_run=True)

    after = _capture_workspace_state(ws)
    # No workspace mutation.
    assert after["page_text"] == before["page_text"], "page bytes changed"
    assert after["fingerprints_text"] == before["fingerprints_text"], (
        "fingerprint store changed"
    )
    assert after["lock_exists"] == before["lock_exists"], (
        "lock file appeared or disappeared"
    )
    assert after["journal_files"] == before["journal_files"], (
        "journal entries changed"
    )

    # The plan describes the shared page with the would-change flags.
    assert result.dry_run is True
    assert result.list_targets is False
    assert result.rendered_pages == []
    assert result.unchanged_pages == []
    assert result.fingerprints == {}
    assert len(result.plan) == 1
    entry = result.plan[0]
    assert isinstance(entry, RenderPlanEntry)
    assert entry.target == "page:concept/shared"
    assert entry.page_id == "concept/shared"
    assert entry.page_path.endswith("shared.md")
    assert entry.block_id == "claim_block.concept.shared"
    assert entry.marker_health == "ok"
    assert entry.content_would_change is False, (
        "ingest already rendered the page; bytes match"
    )
    assert entry.fingerprint_would_change is True, (
        "corrupted stored fingerprint must drive fingerprint_would_change=True"
    )
    assert entry.stored_fingerprint == "sha256:deadbeef"
    assert entry.planned_fingerprint is not None
    assert entry.planned_fingerprint.startswith("sha256:")


def test_dry_run_plan_matches_subsequent_real_render(tmp_path: Path) -> None:
    """Two opposite workspaces with the same canonical state must
    produce the same plan from dry-run and the same outcomes from
    real render. The dry-run plan's would-change flags must match
    what real render actually does.
    """
    ws_plan = _seed_two_entity_workspace(tmp_path / "plan")
    ws_run = _seed_two_entity_workspace(tmp_path / "run")
    # Corrupt the stored fingerprint identically in both workspaces.
    FingerprintStore(ws_plan).set("concept/shared", "sha256:deadbeef")
    FingerprintStore(ws_run).set("concept/shared", "sha256:deadbeef")

    plan_result = render(ws_plan, dry_run=True)
    run_result = render(ws_run)

    assert len(plan_result.plan) == 1
    plan_entry = plan_result.plan[0]

    # The real render rewrites the fingerprint to the planned value.
    assert run_result.fingerprints == {
        plan_entry.page_id: plan_entry.planned_fingerprint
    }
    # The page was unchanged (bytes already match canonical) but the
    # fingerprint advanced. Real render and dry-run agree on both.
    assert any(
        p.endswith("shared.md") for p in run_result.unchanged_pages
    ), run_result
    assert plan_entry.content_would_change is False
    assert plan_entry.fingerprint_would_change is True


def test_list_targets_is_read_only_and_useful(tmp_path: Path) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    before = _capture_workspace_state(ws)

    result = render(ws, list_targets=True)

    after = _capture_workspace_state(ws)
    assert after == before, "list-targets must not mutate the workspace"

    assert result.list_targets is True
    assert result.dry_run is False
    assert result.rendered_pages == []
    assert result.unchanged_pages == []
    assert result.fingerprints == {}
    assert len(result.plan) == 1
    entry = result.plan[0]
    assert entry.target == "page:concept/shared"
    assert entry.page_id == "concept/shared"
    assert entry.page_path.endswith("shared.md")
    assert entry.block_id == "claim_block.concept.shared"
    assert entry.marker_health == "ok"
    # Both entities are listed in entity_id order.
    contributor_ids = [c.entity_id for c in entry.contributors]
    assert contributor_ids == ["concept.alpha", "concept.beta"]
    # Both display names surface.
    assert {c.display_name for c in entry.contributors} == {"Alpha", "Beta"}
    # All claim ids surface, in (entity_id, claim_id) order via the flat list.
    assert entry.contributing_claim_ids == ["c.alpha.1", "c.beta.1"]


def test_target_filtering_and_unknown_target_refusal(tmp_path: Path) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    # Add a lonely page on disk with no claim contributors.
    lonely = ws.pages / "concepts" / "lonely.md"
    lonely.write_text(PAGE_TEMPLATE_LONELY, encoding="utf-8")
    before = _capture_workspace_state(ws)

    # Dry-run with a specific target only returns that page.
    dry = render(ws, target="page:concept/shared", dry_run=True)
    assert [e.page_id for e in dry.plan] == ["concept/shared"]

    # List-targets with the lonely page still surfaces it (page exists
    # on disk but no claim contributors).
    lonely_result = render(ws, target="page:concept/lonely", list_targets=True)
    assert len(lonely_result.plan) == 1
    lonely_entry = lonely_result.plan[0]
    assert lonely_entry.page_id == "concept/lonely"
    assert lonely_entry.marker_health == "ok"
    assert lonely_entry.contributors == []
    assert lonely_entry.contributing_claim_ids == []

    # Unknown target raises lockless ValueError on dry-run AND list-targets.
    for mode in ({"dry_run": True}, {"list_targets": True}):
        with pytest.raises(ValueError) as excinfo:
            render(ws, target="concept/missing", **mode)
        assert "concept/missing" in str(excinfo.value)

    after = _capture_workspace_state(ws)
    assert after == before, "target filtering must not mutate the workspace"


def test_marker_health_parse_error_dry_run_does_not_fail_hard(
    tmp_path: Path,
) -> None:
    """A page with malformed markers reports ``parse_error`` on the
    read-only path; the mutating render still fails hard on the same
    fixture.
    """
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "alpha.md"
    src.write_text(SOURCE_ALPHA, encoding="utf-8")
    # Seed an ingest pointing at the malformed page so it appears in
    # the render plan.
    bad_page = ws.pages / "concepts" / "malformed.md"
    bad_page.parent.mkdir(parents=True, exist_ok=True)
    # Ingest renders against the page; we need the page to exist with
    # working markers first, then break it. Start with the good
    # template, ingest, then corrupt.
    good = PAGE_TEMPLATE.replace("concept/shared", "concept/malformed").replace(
        "claim_block.concept.shared", "claim_block.concept.malformed"
    ).replace("commentary.concept.shared", "commentary.concept.malformed")
    bad_page.write_text(good, encoding="utf-8")
    ingest(
        ws,
        src,
        source_id="src.alpha",
        source_class="markdown_prose",
        seed_claims=[
            SeedClaim(
                entity_id="concept.alpha",
                entity_type="concept",
                display_name="Alpha",
                claim_id="c.alpha.1",
                claim_kind="definition",
                claim_text=(
                    "Alpha entity contributes the first verifiable claim "
                    "about the shared block."
                ),
                locator=Locator(
                    locator_type="markdown_prose_v1",
                    heading_path=["Methods"],
                    paragraph_index=1,
                    sentence_start=1,
                    sentence_end=1,
                ),
                render_target=(
                    "concept/malformed",
                    "claim_block.concept.malformed",
                ),
            )
        ],
    )
    # Now corrupt the markers.
    bad_page.write_text(PAGE_TEMPLATE_MALFORMED, encoding="utf-8")

    # Dry-run surfaces marker_health=parse_error with a message.
    dry = render(ws, target="page:concept/malformed", dry_run=True)
    assert len(dry.plan) == 1
    entry = dry.plan[0]
    assert entry.marker_health == "parse_error"
    assert entry.marker_message
    assert entry.content_would_change is None
    assert entry.fingerprint_would_change is None

    # List-targets reports the same shape (read-only).
    lst = render(ws, target="page:concept/malformed", list_targets=True)
    assert lst.plan[0].marker_health == "parse_error"

    # Real render still fails hard on the same page.
    from llloom.pages.render import RenderError

    with pytest.raises(RenderError):
        render(ws, target="page:concept/malformed")


def test_cli_dry_run_and_list_targets_emit_structured_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    FingerprintStore(ws).set("concept/shared", "sha256:deadbeef")
    before = _capture_workspace_state(ws)

    rc = cli_main(["--root", str(tmp_path), "render", "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["dry_run"] is True
    assert payload["list_targets"] is False
    assert payload["rendered_pages"] == []
    assert payload["unchanged_pages"] == []
    assert payload["fingerprints"] == {}
    assert isinstance(payload["plan"], list) and len(payload["plan"]) == 1
    entry = payload["plan"][0]
    assert entry["target"] == "page:concept/shared"
    assert entry["page_id"] == "concept/shared"
    assert entry["page_path"].endswith("shared.md")
    assert entry["block_id"] == "claim_block.concept.shared"
    assert entry["marker_health"] == "ok"
    assert entry["content_would_change"] is False
    assert entry["fingerprint_would_change"] is True
    assert entry["stored_fingerprint"] == "sha256:deadbeef"
    assert entry["planned_fingerprint"].startswith("sha256:")
    assert {c["entity_id"] for c in entry["contributors"]} == {
        "concept.alpha",
        "concept.beta",
    }
    assert entry["contributing_claim_ids"] == ["c.alpha.1", "c.beta.1"]

    # Workspace unchanged after the CLI dry-run.
    after_dry = _capture_workspace_state(ws)
    assert after_dry == before

    # CLI list-targets payload shape.
    rc = cli_main(["--root", str(tmp_path), "render", "--list-targets"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["list_targets"] is True
    assert payload["dry_run"] is False
    assert len(payload["plan"]) == 1
    assert payload["plan"][0]["target"] == "page:concept/shared"

    # Workspace still unchanged after the CLI list-targets.
    after_list = _capture_workspace_state(ws)
    assert after_list == before

    # CLI exits non-zero on unknown target without mutating anything.
    rc = cli_main(
        ["--root", str(tmp_path), "render", "--dry-run", "concept/missing"]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "Traceback" not in captured.err
    assert "page:concept/missing" in captured.err
    after_err = _capture_workspace_state(ws)
    assert after_err == before, (
        "unknown-target dry-run refusal must not mutate the workspace"
    )
