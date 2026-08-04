"""Contract: query filters, ids-only, claim cards, and lists (Slice 077).

Pins the read-only authority-visibility surface added on top of the
endorsed Slice 076 seed-apply path. Every operation here is read-only
— no lock, no journal, no model call, no transaction directory, no
page / fingerprint / claim / source / sidecar / report write.

The slice also pins five new top-level CLI verbs: ``list-claims``,
``claim-card``, ``list-sources``, ``list-pages``,
``list-render-targets``. The verb-count guard at
``tests/contract/test_prepare_pdf_cli.py`` is updated separately.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from llloom.claims.models import Locator
from llloom.claims.store import ClaimStore
from llloom.cli import main as cli_main
from llloom.llm.harness import LLMInvoke
from llloom.ops import (
    claim_card,
    list_claims,
    list_pages,
    list_render_targets,
    list_sources,
    query,
)
from llloom.ops.inspect import ClaimCardError
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.rebuild import rebuild
from llloom.ops.results import ClaimCard, ClaimSummary, SourceSummary
from llloom.state.search import sidecar_exists
from llloom.workspace.layout import Workspace


SOURCE_TEXT = """\
# Article

## Methods

Alpha is documented in the source. It anchors the inspection slice.

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


def _seed_claims(ws: Workspace) -> None:
    seeds = [
        SeedClaim(
            entity_id="concept.alpha",
            entity_type="concept",
            display_name="Alpha",
            claim_id="c.alpha.1",
            claim_kind="definition",
            claim_text="Alpha is documented in the source. It anchors the inspection slice.",
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
            entity_id="concept.beta",
            entity_type="concept",
            display_name="Beta",
            claim_id="c.beta.1",
            claim_kind="property",
            claim_text="Beta is a separate sentence about a different entity.",
            locator=Locator(
                locator_type="markdown_prose_v1",
                heading_path=["Methods"],
                paragraph_index=2,
                sentence_start=1,
                sentence_end=1,
            ),
            render_target=("concept/beta", "claim_block.concept.beta"),
        ),
    ]
    ingest(
        ws,
        ws.raw_sources / "article.md",
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=seeds,
    )


def _mutate_claim_status(
    ws: Workspace, entity_id: str, claim_id: str, *, status: str
) -> None:
    """Direct YAML mutation so a fixture can place a claim in an
    arbitrary lifecycle state without exercising the ``promote`` path.
    The slice-under-test only reads; this fixture write happens
    **before** any list/card/query call.
    """
    path = ws.claims_entities / f"{entity_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for assertion in data.get("assertions", []):
        if assertion.get("claim_id") == claim_id:
            assertion["status"] = status
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _mutate_claim_field(
    ws: Workspace,
    entity_id: str,
    claim_id: str,
    *,
    field: str,
    value: str,
) -> None:
    path = ws.claims_entities / f"{entity_id}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for assertion in data.get("assertions", []):
        if assertion.get("claim_id") == claim_id:
            assertion[field] = value
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _snapshot_state(ws: Workspace) -> dict[str, object]:
    """Capture every file path + content hash + presence flag the
    read-only operations are required to leave byte-identical.
    """
    state: dict[str, object] = {}
    # Claim YAML files.
    state["claims"] = {
        p.name: p.read_bytes()
        for p in sorted(ws.claims_entities.glob("*.yaml"))
    }
    # Source registry records.
    state["sources"] = {
        p.name: p.read_bytes()
        for p in sorted(ws.state_source_registry.glob("*.yaml"))
    }
    # Page files.
    state["pages"] = {
        str(p.relative_to(ws.pages)): p.read_bytes()
        for p in sorted(ws.pages.rglob("*.md"))
    }
    # Render fingerprints.
    state["fingerprints"] = (
        ws.render_fingerprints.read_bytes()
        if ws.render_fingerprints.is_file()
        else None
    )
    # Lock file.
    state["lock_present"] = (ws.state_locks / "workspace.yaml").is_file()
    # Journal entries.
    state["journals"] = sorted(p.name for p in ws.state_journals.glob("*.yaml"))
    # Transactions.
    state["transactions"] = sorted(
        str(p.relative_to(ws.state_transactions))
        for p in ws.state_transactions.rglob("*")
    )
    # Seed update reports.
    state["seed_reports"] = sorted(
        p.name for p in ws.state_reports_updates.glob("*.yaml")
    )
    # Search sidecar present?
    state["search_present"] = ws.search_db.is_file()
    return state


# ---------------------------------------------------------------------
# 1. Status filter: default skips, --status all surfaces all,
#    explicit --status draft / superseded works in either direction.
# ---------------------------------------------------------------------


def test_status_filter_default_skips_and_all_surfaces(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)
    _mutate_claim_status(ws, "concept.beta", "c.beta.1", status="superseded")

    # Default behavior: superseded claim must NOT appear in query
    # citations even when the text matches.
    default_result = query(ws, question="Beta")
    default_ids = [c["claim_id"] for c in default_result.citations]
    assert "c.beta.1" not in default_ids

    # Explicit status=superseded surfaces the superseded claim.
    super_result = query(ws, question="Beta", status="superseded")
    super_ids = [c["claim_id"] for c in super_result.citations]
    assert super_ids == ["c.beta.1"]

    # status="all" surfaces every state (alpha=draft + beta=superseded).
    all_result = query(ws, question="", status="all", max_citations=10)
    all_ids = {c["claim_id"] for c in all_result.citations}
    assert {"c.alpha.1", "c.beta.1"}.issubset(all_ids)

    # Explicit --status draft returns only drafts.
    draft_result = query(ws, question="", status="draft", max_citations=10)
    draft_ids = {c["claim_id"] for c in draft_result.citations}
    assert "c.alpha.1" in draft_ids
    assert "c.beta.1" not in draft_ids


# ---------------------------------------------------------------------
# 2. Verification status filter composes with text query.
# ---------------------------------------------------------------------


def test_verification_status_filter_composes(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)
    # Ingest stamps both claims as ``verified``; flip beta to
    # ``unverified`` for the filter test.
    _mutate_claim_field(
        ws, "concept.beta", "c.beta.1", field="verification_status", value="unverified"
    )

    verified_result = query(
        ws, question="", verification_status="verified", max_citations=10
    )
    verified_ids = {c["claim_id"] for c in verified_result.citations}
    assert verified_ids == {"c.alpha.1"}

    unverified_result = query(
        ws, question="", verification_status="unverified", max_citations=10
    )
    unverified_ids = {c["claim_id"] for c in unverified_result.citations}
    assert unverified_ids == {"c.beta.1"}


# ---------------------------------------------------------------------
# 3. Role/kind and entity filters compose with the text query.
# ---------------------------------------------------------------------


def test_role_and_entity_filters_compose(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)

    only_alpha = query(ws, question="", entity_id="concept.alpha", max_citations=10)
    assert {c["entity_id"] for c in only_alpha.citations} == {"concept.alpha"}

    only_definitions = query(
        ws, question="", role="definition", max_citations=10
    )
    assert {c["claim_kind"] for c in only_definitions.citations} == {"definition"}

    # AND composition: definition role AND beta entity yields nothing
    # (beta is a property, alpha is a definition).
    composed = query(
        ws,
        question="",
        role="definition",
        entity_id="concept.beta",
        max_citations=10,
    )
    assert composed.citations == []


# ---------------------------------------------------------------------
# 4. ids-only emits stable ids with no extra prose.
# ---------------------------------------------------------------------


def test_query_ids_only_cli_emits_one_id_per_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)

    # Library: ids_only result has empty answer + populated used_claim_ids.
    result = query(ws, question="Alpha", ids_only=True)
    assert result.ids_only is True
    assert result.answer == ""
    assert "c.alpha.1" in result.used_claim_ids

    # CLI: --ids-only emits one id per line, no JSON envelope, no
    # bullets, no trailing commentary.
    rc = cli_main(
        ["--root", str(tmp_path), "query", "Alpha", "--ids-only"]
    )
    captured = capsys.readouterr()
    assert rc == 0
    lines = [line for line in captured.out.split("\n") if line]
    assert lines, "ids-only must print at least one id"
    for line in lines:
        # Each line must be a bare claim id — no JSON / no bullets.
        assert not line.startswith("{")
        assert not line.startswith("- ")
        # No prose envelope.
        assert "answer" not in line.lower()
    assert "c.alpha.1" in lines


# ---------------------------------------------------------------------
# 5. Query citations foreground authority metadata.
# ---------------------------------------------------------------------


def test_citations_carry_authority_metadata(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)

    result = query(ws, question="Alpha")
    assert result.citations, "expected at least one citation"
    citation = result.citations[0]
    # Required fields.
    for field in (
        "entity_id",
        "claim_id",
        "claim_kind",
        "status",
        "verification_status",
        "source_ids",
        "render_targets",
    ):
        assert field in citation, f"citation missing {field!r}: {citation}"
    assert isinstance(citation["source_ids"], list)
    assert "src.article" in citation["source_ids"]
    assert isinstance(citation["render_targets"], list)
    rt = citation["render_targets"][0]
    assert rt == {
        "page_id": "concept/alpha",
        "block_id": "claim_block.concept.alpha",
    }


# ---------------------------------------------------------------------
# 6. Claim card surfaces canonical fields; bare ambiguous ids refuse.
# ---------------------------------------------------------------------


def test_claim_card_qualified_and_bare_and_ambiguous(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)
    # Add a second claim with the same bare claim_id under a
    # *different* entity to force ambiguity for the bare-id refusal.
    second_seed = SeedClaim(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        claim_id="c.dup",
        claim_kind="definition",
        claim_text="Alpha is documented in the source. It anchors the inspection slice.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=2,
        ),
        render_target=("concept/alpha", "claim_block.concept.alpha"),
    )
    third_seed = SeedClaim(
        entity_id="concept.beta",
        entity_type="concept",
        display_name="Beta",
        claim_id="c.dup",
        claim_kind="property",
        claim_text="Beta is a separate sentence about a different entity.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=2,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("concept/beta", "claim_block.concept.beta"),
    )
    ingest(
        ws,
        ws.raw_sources / "article.md",
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[second_seed, third_seed],
    )

    # Qualified form resolves.
    card = claim_card(ws, "claim:concept.alpha:c.alpha.1")
    assert isinstance(card, ClaimCard)
    assert card.qualified_target == "claim:concept.alpha:c.alpha.1"
    assert card.claim_id == "c.alpha.1"
    assert card.entity_id == "concept.alpha"
    assert card.entity_display_name == "Alpha"
    assert card.entity_type == "concept"
    assert card.claim_kind == "definition"
    assert card.status == "draft"
    assert card.verification_status == "verified"
    assert card.render_targets[0].page_id == "concept/alpha"
    assert card.render_targets[0].block_id == "claim_block.concept.alpha"
    assert card.evidence, "card must surface at least one evidence summary"
    assert card.evidence[0].source_id == "src.article"
    assert card.evidence[0].excerpt_hash.startswith("sha256:")

    # Unique bare id still resolves.
    bare_card = claim_card(ws, "c.alpha.1")
    assert bare_card.qualified_target == "claim:concept.alpha:c.alpha.1"

    # Ambiguous bare id refuses with both candidates named.
    with pytest.raises(ClaimCardError) as excinfo:
        claim_card(ws, "c.dup")
    msg = str(excinfo.value)
    assert "ambiguous" in msg
    assert "claim:concept.alpha:c.dup" in msg
    assert "claim:concept.beta:c.dup" in msg

    # Unknown bare id refuses.
    with pytest.raises(ClaimCardError):
        claim_card(ws, "c.does-not-exist")


# ---------------------------------------------------------------------
# 7. All list / card / query commands are read-only.
# ---------------------------------------------------------------------


def test_list_and_card_commands_are_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)
    # Monkeypatch LLMInvoke.invoke to raise so any accidental model
    # call surfaces immediately.
    monkeypatch.setattr(
        LLMInvoke,
        "invoke",
        lambda self, *a, **kw: (_ for _ in ()).throw(
            AssertionError("LLMInvoke.invoke was called from a read-only path")
        ),
    )

    before = _snapshot_state(ws)

    # Exercise every new surface + the extended query path.
    _ = query(
        ws, question="Alpha", status="all", verification_status="verified"
    )
    _ = list_claims(ws)
    _ = claim_card(ws, "claim:concept.alpha:c.alpha.1")
    _ = list_sources(ws)
    _ = list_pages(ws)
    _ = list_render_targets(ws)

    after = _snapshot_state(ws)
    assert before == after, (
        "read-only operations mutated the workspace: "
        f"diff = "
        f"{ {k: (before[k], after[k]) for k in before if before[k] != after[k]} }"
    )


# ---------------------------------------------------------------------
# 8. ids-only output is deterministic across all list verbs.
# ---------------------------------------------------------------------


def test_list_commands_ids_only_is_stable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)

    def _run(*argv: str) -> list[str]:
        rc = cli_main(["--root", str(tmp_path), *argv, "--ids-only"])
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        return [line for line in captured.out.split("\n") if line]

    claims = _run("list-claims")
    assert claims == sorted(claims)
    assert "claim:concept.alpha:c.alpha.1" in claims
    assert "claim:concept.beta:c.beta.1" in claims

    sources = _run("list-sources")
    assert "src.article" in sources

    pages = _run("list-pages")
    assert "concept/alpha" in pages
    assert "concept/beta" in pages

    targets = _run("list-render-targets")
    assert "concept/alpha" in targets
    assert "concept/beta" in targets

    # Determinism: each command emits the same ids on a second call.
    claims2 = _run("list-claims")
    assert claims2 == claims


# ---------------------------------------------------------------------
# 9. Sidecar cannot bypass canonical filters.
# ---------------------------------------------------------------------


def test_sidecar_cannot_bypass_canonical_filters(tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)
    # Build a search sidecar so the sidecar path runs.
    rebuild(ws, target="search")
    assert sidecar_exists(ws)

    # Mutate the canonical YAML so beta's status becomes 'retracted'.
    # The sidecar still indexes the original ``draft`` row — it has not
    # been rebuilt. The canonical filter must still drop beta from a
    # default query.
    _mutate_claim_status(ws, "concept.beta", "c.beta.1", status="retracted")

    result = query(ws, question="Beta")
    citation_ids = [c["claim_id"] for c in result.citations]
    assert "c.beta.1" not in citation_ids

    # And an explicit ``--verification-status=verified`` query that
    # also excludes the still-verified alpha by a different filter
    # path (entity=beta) must return nothing — the sidecar must not
    # smuggle the retracted row through.
    composed = query(
        ws, question="Beta", entity_id="concept.beta", status="all"
    )
    # Now beta IS surfaced because status='all' opts into retracted
    # too. The point is that the canonical YAML's current status field
    # ('retracted') drives the answer, not the stale sidecar row.
    surfaced = [c for c in composed.citations if c["claim_id"] == "c.beta.1"]
    assert surfaced
    assert surfaced[0]["status"] == "retracted"


# ---------------------------------------------------------------------
# Slice 077a follow-up: default empty-query compatibility cleanup.
#
# Slice 077 accidentally widened the plain ``query("")`` path because
# ``_retrieve_claims`` admitted ``not tokens`` unconditionally. The
# cleanup gates empty-token admission behind explicit inspection — any
# Slice 077 filter knob (status / verification_status / entity_id /
# role) or ``ids_only=True``. The plain default ``query("")`` is once
# again byte-compatible with the pre-Slice-077 behavior.
# ---------------------------------------------------------------------


def test_default_empty_query_returns_no_citations(tmp_path: Path) -> None:
    """Slice 077a: plain default ``query("")`` returns no citations,
    no used ids, and the deterministic no-match answer — same as
    pre-Slice-077.
    """
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)

    result = query(ws, question="")
    assert result.citations == []
    assert result.used_claim_ids == []
    assert "No authoritative claims" in result.answer
    assert result.ids_only is False


def test_explicit_filtered_empty_query_still_works(tmp_path: Path) -> None:
    """Slice 077a: empty question + explicit filter still surfaces
    matching claims (the inspection-through-query behavior Slice 077
    added remains intact).
    """
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)

    # status="all" opts into broad inspection.
    all_result = query(ws, question="", status="all", max_citations=10)
    assert {c["claim_id"] for c in all_result.citations} == {"c.alpha.1", "c.beta.1"}

    # status="draft" narrows.
    draft_result = query(ws, question="", status="draft", max_citations=10)
    assert {c["claim_id"] for c in draft_result.citations} == {"c.alpha.1", "c.beta.1"}

    # verification_status filter alone also counts as explicit inspection.
    verified_result = query(
        ws, question="", verification_status="verified", max_citations=10
    )
    assert {c["claim_id"] for c in verified_result.citations} == {
        "c.alpha.1",
        "c.beta.1",
    }

    # entity_id filter alone also counts as explicit inspection.
    only_alpha = query(
        ws, question="", entity_id="concept.alpha", max_citations=10
    )
    assert {c["entity_id"] for c in only_alpha.citations} == {"concept.alpha"}

    # role filter alone also counts as explicit inspection.
    only_definitions = query(
        ws, question="", role="definition", max_citations=10
    )
    assert {c["claim_kind"] for c in only_definitions.citations} == {"definition"}


def test_cli_default_empty_query_emits_empty_citations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Slice 077a: ``llloom query ""`` emits a normal ``QueryResult``
    JSON whose ``citations`` and ``used_claim_ids`` are both empty.
    """
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)

    rc = cli_main(["--root", str(tmp_path), "query", ""])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["citations"] == []
    assert payload["used_claim_ids"] == []
    assert payload["ids_only"] is False


def test_ids_only_empty_query_counts_as_explicit_inspection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Slice 077a chose to treat ``ids_only=True`` as explicit
    inspection so a shell user running ``llloom query --ids-only ""``
    gets the obvious "list every default-visible id" behavior without
    having to pair the flag with ``--status all``.

    The pre-existing
    ``test_query_ids_only_cli_emits_one_id_per_line`` covers the
    non-empty-question path; this test pins the empty-question path
    end-to-end at both the library and CLI surfaces.
    """
    ws = _seed_workspace(tmp_path)
    _seed_claims(ws)

    # Library path.
    result = query(ws, question="", ids_only=True, max_citations=10)
    assert result.ids_only is True
    assert result.answer == ""
    assert set(result.used_claim_ids) == {"c.alpha.1", "c.beta.1"}

    # CLI path.
    rc = cli_main(["--root", str(tmp_path), "query", "", "--ids-only"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    lines = [line for line in captured.out.split("\n") if line]
    assert set(lines) == {"c.alpha.1", "c.beta.1"}
    for line in lines:
        assert not line.startswith("{")
        assert not line.startswith("- ")
