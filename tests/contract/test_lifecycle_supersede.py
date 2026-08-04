"""Contract: lifecycle state machine + direct supersede (Slice 078).

Pins:

- the legal lifecycle transition graph lives in one place
  (``llloom.claims.lifecycle``);
- ``promote(...)`` consults the shared guard (no private transition
  table drift);
- ``supersede(workspace, *, old, by)`` flips the OLD claim to
  ``superseded``, records a supersession link on the NEW claim,
  refuses unsafe states without mutation, and is atomic under
  ``operation(...)``;
- bare-id ambiguity / missing-id resolution is safe and operator-
  friendly;
- the CLI verb is wired and the verb-count guard is updated;
- the read / query / list / card surfaces from Slices 077 / 077a
  remain coherent for superseded claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llloom.claims.lifecycle import (
    LEGAL_LIFECYCLE_TRANSITIONS,
    SOURCE_CASCADE_STATUSES,
    can_transition,
    explain_transition_refusal,
)
from llloom.claims.models import CLAIM_STATUSES, Locator
from llloom.claims.store import ClaimStore
from llloom.cli import main as cli_main
from llloom.ops import claim_card, list_claims, promote, query, supersede
from llloom.ops.results import SupersedeResult
from llloom.ops.supersede import SupersedeError
from llloom.ops.ingest import SeedClaim, ingest
from llloom.state.journal import OperationJournal
from llloom.workspace.layout import Workspace


SOURCE_TEXT = """\
# Article

## Methods

Alpha is documented in the source. It anchors the supersede slice.

Beta is a separate sentence about a different entity.
"""


PAGE_ALPHA = """\
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


PAGE_BETA = """\
---
page_id: concept/beta
page_class: concept
write_policy: mixed
status: rendered
---

<!-- llloom:claim-block id=claim_block.concept.beta -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.beta owner=human -->
Commentary beta.
<!-- /llloom:commentary -->
"""


def _seed_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    (ws.pages / "concepts").mkdir(parents=True, exist_ok=True)
    (ws.pages / "concepts" / "alpha.md").write_text(PAGE_ALPHA, encoding="utf-8")
    (ws.pages / "concepts" / "beta.md").write_text(PAGE_BETA, encoding="utf-8")
    return ws


def _seed_two_claims(ws: Workspace) -> None:
    seeds = [
        SeedClaim(
            entity_id="concept.alpha",
            entity_type="concept",
            display_name="Alpha",
            claim_id="c.alpha.1",
            claim_kind="definition",
            claim_text="Alpha is documented in the source. It anchors the supersede slice.",
            locator=Locator(
                locator_type="markdown_prose_v1",
                heading_path=["Methods"],
                paragraph_index=1,
                sentence_start=1,
                sentence_end=2,
            ),
            render_target=("concept/alpha", "claim_block.concept.alpha"),
        ),
        SeedClaim(
            entity_id="concept.alpha",
            entity_type="concept",
            display_name="Alpha",
            claim_id="c.alpha.2",
            claim_kind="definition",
            claim_text="Alpha is documented in the source. It anchors the supersede slice.",
            locator=Locator(
                locator_type="markdown_prose_v1",
                heading_path=["Methods"],
                paragraph_index=1,
                sentence_start=1,
                sentence_end=2,
            ),
            render_target=("concept/alpha", "claim_block.concept.alpha"),
        ),
    ]
    ingest(
        ws,
        ws.raw_sources / "article.md",
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=seeds,
    )


def _force_status(
    ws: Workspace, entity_id: str, claim_id: str, *, status: str
) -> None:
    """Direct YAML mutation so a fixture can put a claim into an
    arbitrary lifecycle state without exercising the ``promote``
    path. The slice-under-test still walks the canonical YAML once
    the fixture is in place.
    """
    path = ws.claims_entities / f"{entity_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for assertion in data.get("assertions", []):
        if assertion.get("claim_id") == claim_id:
            assertion["status"] = status
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _validated_pair(ws: Workspace) -> None:
    _seed_two_claims(ws)
    _force_status(ws, "concept.alpha", "c.alpha.1", status="validated")
    _force_status(ws, "concept.alpha", "c.alpha.2", status="validated")


def _journal_count(ws: Workspace, op_kind: str | None = None) -> int:
    if not ws.state_journals.is_dir():
        return 0
    total = 0
    for path in ws.state_journals.glob("*.yaml"):
        if op_kind is None:
            total += 1
        else:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if payload.get("op_kind") == op_kind:
                total += 1
    return total


# ---------------------------------------------------------------------
# 1. Lifecycle helper legal graph
# ---------------------------------------------------------------------


def test_lifecycle_helper_legal_graph() -> None:
    expected = {
        ("draft", "reviewed"),
        ("reviewed", "validated"),
        ("validated", "superseded"),
        ("validated", "archived"),
    }
    assert LEGAL_LIFECYCLE_TRANSITIONS == frozenset(expected)
    for from_status, to_status in expected:
        assert can_transition(from_status, to_status)

    # Reverse edges, self-edges, and skip-edges are all illegal.
    assert not can_transition("reviewed", "draft")
    assert not can_transition("draft", "draft")
    assert not can_transition("draft", "validated")
    assert not can_transition("draft", "superseded")

    # Source-cascade statuses are never operator-promotable.
    assert SOURCE_CASCADE_STATUSES == frozenset(
        {"retracted", "retracted_by_source", "stale"}
    )
    for cascade in SOURCE_CASCADE_STATUSES:
        for target in CLAIM_STATUSES:
            if target == cascade:
                continue
            assert not can_transition(cascade, target), (
                f"{cascade} -> {target} must not be operator-promotable"
            )
        msg = explain_transition_refusal(cascade, "reviewed")
        assert "source-cascade" in msg
        assert cascade in msg

    # Refusal messages are actionable.
    msg = explain_transition_refusal("draft", "validated")
    assert "draft" in msg and "validated" in msg
    assert "'reviewed'" in msg  # suggests intermediate step
    self_msg = explain_transition_refusal("validated", "validated")
    assert "already" in self_msg


# ---------------------------------------------------------------------
# 2. Promote uses the centralized lifecycle guard
# ---------------------------------------------------------------------


def test_promote_uses_centralized_lifecycle_guard(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_two_claims(ws)
    # Force c.alpha.1 to 'reviewed' so a draft->validated jump becomes
    # a reviewed->validated promotion that still goes through the
    # shared guard.

    # Refused: draft -> validated (skip edge).
    bad = promote(
        ws,
        target="claim:concept.alpha:c.alpha.1",
        to_status="validated",
    )
    assert bad.refused is True
    assert "'reviewed'" in (bad.reason or "")

    # Legal path: draft -> reviewed.
    good1 = promote(
        ws,
        target="claim:concept.alpha:c.alpha.1",
        to_status="reviewed",
    )
    assert good1.refused is False
    assert good1.to_status == "reviewed"

    # Legal path: reviewed -> validated.
    good2 = promote(
        ws,
        target="claim:concept.alpha:c.alpha.1",
        to_status="validated",
    )
    assert good2.refused is False
    assert good2.to_status == "validated"


# ---------------------------------------------------------------------
# 3. Direct supersede succeeds atomically
# ---------------------------------------------------------------------


def test_supersede_succeeds_atomically(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _validated_pair(ws)

    result = supersede(
        ws,
        old="claim:concept.alpha:c.alpha.1",
        by="claim:concept.alpha:c.alpha.2",
    )
    assert isinstance(result, SupersedeResult)
    assert result.refused is False
    assert result.old_target == "claim:concept.alpha:c.alpha.1"
    assert result.new_target == "claim:concept.alpha:c.alpha.2"
    assert result.old_from_status == "validated"
    assert result.old_to_status == "superseded"
    assert result.new_status == "validated"
    assert result.supersedes == "claim:concept.alpha:c.alpha.1"
    assert result.op_id

    # Canonical YAML reflects both changes.
    entity = ClaimStore(ws).load_entity("concept.alpha")
    old_a = entity.find_assertion("c.alpha.1")
    new_a = entity.find_assertion("c.alpha.2")
    assert old_a is not None and old_a.status == "superseded"
    assert new_a is not None and new_a.status == "validated"
    assert new_a.supersedes == "claim:concept.alpha:c.alpha.1"

    # Exactly one supersede journal entry exists with the touched
    # entity path recorded.
    assert _journal_count(ws, op_kind="supersede") == 1
    entry = OperationJournal(ws).load(result.op_id)
    assert entry.op_kind == "supersede"
    assert any("concept.alpha.yaml" in tf for tf in entry.touched_files)


# ---------------------------------------------------------------------
# 4. Direct supersede refuses unsafe states without mutation
# ---------------------------------------------------------------------


def test_supersede_refuses_unsafe_states_without_mutation(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_two_claims(ws)
    # OLD draft (illegal source state for the lifecycle guard),
    # NEW also default draft (would also fail the validated-replacement
    # rule).
    yaml_before = (ws.claims_entities / "concept.alpha.yaml").read_bytes()

    draft_old = supersede(
        ws,
        old="claim:concept.alpha:c.alpha.1",
        by="claim:concept.alpha:c.alpha.2",
    )
    assert draft_old.refused is True
    assert "draft" in (draft_old.reason or "")
    # No mutation.
    assert (ws.claims_entities / "concept.alpha.yaml").read_bytes() == yaml_before

    # Same-claim refusal.
    _force_status(ws, "concept.alpha", "c.alpha.1", status="validated")
    _force_status(ws, "concept.alpha", "c.alpha.2", status="validated")
    yaml_before = (ws.claims_entities / "concept.alpha.yaml").read_bytes()
    self_super = supersede(
        ws,
        old="claim:concept.alpha:c.alpha.1",
        by="claim:concept.alpha:c.alpha.1",
    )
    assert self_super.refused is True
    assert "itself" in (self_super.reason or "")
    assert (ws.claims_entities / "concept.alpha.yaml").read_bytes() == yaml_before

    # NEW draft (not validated) refuses with the symmetric-authority
    # rule.
    _force_status(ws, "concept.alpha", "c.alpha.2", status="draft")
    yaml_before = (ws.claims_entities / "concept.alpha.yaml").read_bytes()
    new_not_validated = supersede(
        ws,
        old="claim:concept.alpha:c.alpha.1",
        by="claim:concept.alpha:c.alpha.2",
    )
    assert new_not_validated.refused is True
    assert "validated" in (new_not_validated.reason or "")
    assert (ws.claims_entities / "concept.alpha.yaml").read_bytes() == yaml_before


# ---------------------------------------------------------------------
# 5. Bare-id resolution is safe
# ---------------------------------------------------------------------


def test_bare_id_resolution_safe(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _validated_pair(ws)
    # Plant a duplicate claim_id under a different entity to force
    # ambiguity for the bare-id refusal.
    ingest(
        ws,
        ws.raw_sources / "article.md",
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[
            SeedClaim(
                entity_id="concept.beta",
                entity_type="concept",
                display_name="Beta",
                claim_id="c.alpha.1",
                claim_kind="definition",
                claim_text="Alpha is documented in the source. It anchors the supersede slice.",
                locator=Locator(
                    locator_type="markdown_prose_v1",
                    heading_path=["Methods"],
                    paragraph_index=1,
                    sentence_start=1,
                    sentence_end=2,
                ),
                render_target=("concept/beta", "claim_block.concept.beta"),
            )
        ],
    )

    # Ambiguous OLD bare id.
    with pytest.raises(SupersedeError) as exc:
        supersede(ws, old="c.alpha.1", by="claim:concept.alpha:c.alpha.2")
    msg = str(exc.value)
    assert "ambiguous" in msg
    assert "claim:concept.alpha:c.alpha.1" in msg
    assert "claim:concept.beta:c.alpha.1" in msg

    # Missing bare id.
    with pytest.raises(SupersedeError) as exc:
        supersede(
            ws,
            old="claim:concept.alpha:c.alpha.1",
            by="c.does-not-exist",
        )
    assert "no claim" in str(exc.value)

    # Unique bare id resolves. Use c.alpha.2 which is unique
    # (c.alpha.1 was duplicated above).
    yaml_before = (ws.claims_entities / "concept.alpha.yaml").read_bytes()
    # We need a fresh validated replacement; create one with a unique id.
    ingest(
        ws,
        ws.raw_sources / "article.md",
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[
            SeedClaim(
                entity_id="concept.alpha",
                entity_type="concept",
                display_name="Alpha",
                claim_id="c.alpha.unique",
                claim_kind="definition",
                claim_text="Alpha is documented in the source. It anchors the supersede slice.",
                locator=Locator(
                    locator_type="markdown_prose_v1",
                    heading_path=["Methods"],
                    paragraph_index=1,
                    sentence_start=1,
                    sentence_end=2,
                ),
                render_target=("concept/alpha", "claim_block.concept.alpha"),
            )
        ],
    )
    _force_status(ws, "concept.alpha", "c.alpha.unique", status="validated")
    # The bare 'c.alpha.unique' must resolve uniquely now.
    result = supersede(ws, old="claim:concept.alpha:c.alpha.2", by="c.alpha.unique")
    assert result.refused is False
    assert result.new_target == "claim:concept.alpha:c.alpha.unique"


# ---------------------------------------------------------------------
# 6. CLI behavior
# ---------------------------------------------------------------------


def test_cli_supersede_success_and_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = _seed_workspace(tmp_path)
    _validated_pair(ws)

    rc = cli_main(
        [
            "--root",
            str(tmp_path),
            "supersede",
            "claim:concept.alpha:c.alpha.1",
            "--by",
            "claim:concept.alpha:c.alpha.2",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["refused"] is False
    assert payload["old_target"] == "claim:concept.alpha:c.alpha.1"
    assert payload["new_target"] == "claim:concept.alpha:c.alpha.2"
    assert payload["old_to_status"] == "superseded"

    # Refusal: try to re-supersede the now-superseded OLD claim.
    rc_bad = cli_main(
        [
            "--root",
            str(tmp_path),
            "supersede",
            "claim:concept.alpha:c.alpha.1",
            "--by",
            "claim:concept.alpha:c.alpha.2",
        ]
    )
    captured_bad = capsys.readouterr()
    assert rc_bad == 1
    payload_bad = json.loads(captured_bad.out)
    assert payload_bad["refused"] is True
    assert payload_bad["reason"]


# ---------------------------------------------------------------------
# 7. Read/query/render inspection remains coherent
# ---------------------------------------------------------------------


def test_query_list_card_remain_coherent_for_superseded(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _validated_pair(ws)
    supersede(
        ws,
        old="claim:concept.alpha:c.alpha.1",
        by="claim:concept.alpha:c.alpha.2",
    )

    # Default query excludes the superseded claim.
    default = query(ws, question="Alpha", max_citations=10)
    default_ids = {c["claim_id"] for c in default.citations}
    assert "c.alpha.1" not in default_ids
    assert "c.alpha.2" in default_ids

    # status="superseded" surfaces the superseded claim.
    super_only = query(ws, question="Alpha", status="superseded", max_citations=10)
    super_ids = {c["claim_id"] for c in super_only.citations}
    assert super_ids == {"c.alpha.1"}

    # status="all" surfaces both.
    all_visible = query(ws, question="", status="all", max_citations=10)
    all_ids = {c["claim_id"] for c in all_visible.citations}
    assert {"c.alpha.1", "c.alpha.2"}.issubset(all_ids)

    # list_claims (default) shows every state including superseded.
    summaries = list_claims(ws)
    statuses = {s.claim_id: s.status for s in summaries}
    assert statuses["c.alpha.1"] == "superseded"
    assert statuses["c.alpha.2"] == "validated"

    # claim_card on the new claim exposes the supersession link.
    card = claim_card(ws, "claim:concept.alpha:c.alpha.2")
    assert card.supersedes == "claim:concept.alpha:c.alpha.1"
    # claim_card on the old claim records its current status.
    old_card = claim_card(ws, "claim:concept.alpha:c.alpha.1")
    assert old_card.status == "superseded"
    assert old_card.supersedes is None
