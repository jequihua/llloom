"""Contract: render fingerprint reconciliation (Slice 072).

Failing-test-first investigation of the WME Audio field-report
symptom that ``lint`` warnings about stale render fingerprints
"survived successful renders" across M009 / M010 / M016. Slice 071
moved every fingerprint-aware surface to a shared page/block-centric
union helper (``compute_page_render_fingerprints`` for `lint` /
`rebuild` / `health_report`; `compute_render_fingerprint_from_contributors`
for `render` / `ingest` / `reconcile` / `retract`). This file pins
the user-facing contract:

- if ``lint`` reports a stale fingerprint for a page, a subsequent
  successful ``render`` (or ``reconcile``) must clear it;
- unchanged content with a correct fingerprint stays clean;
- a pre-write failure must not advance the stored fingerprint past
  the prior known-good value;
- the value ``render`` writes equals the value
  ``compute_page_render_fingerprints`` recomputes for the same
  page, including the multi-entity union case.

Slice 071 + 071a should have already aligned the hash format across
every surface. These tests pin the alignment as regression coverage
so future renderer refactors cannot accidentally reintroduce the
field-report symptom.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.claims.store import ClaimStore
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.lint import lint
from llloom.ops.reconcile import reconcile
from llloom.ops.render import render
from llloom.pages.render import compute_page_render_fingerprints
from llloom.state.fingerprints import FingerprintStore
from llloom.workspace.layout import Workspace


PAGE_TEMPLATE = """\
---
page_id: concept/finger
page_class: concept
write_policy: mixed
status: draft
---

<!-- llloom:claim-block id=claim_block.concept.finger -->
placeholder
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.finger owner=human -->

Commentary that must survive byte-for-byte.

<!-- /llloom:commentary -->
"""


SOURCE_TEXT = """\
# Article

## Methods

The fingerprint test source asserts one verifiable claim about reconciliation.

Second paragraph irrelevant.
"""


SOURCE_TEXT_BETA = """\
# Article

## Methods

The beta entity contributes a second claim to the same shared block.

Second paragraph irrelevant.
"""


def _seed_one_entity_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "alpha.md"
    src.write_text(SOURCE_TEXT, encoding="utf-8")
    page_path = ws.pages / "concepts" / "finger.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    seed = SeedClaim(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        claim_id="c.alpha.1",
        claim_kind="definition",
        claim_text=(
            "The fingerprint test source asserts one verifiable claim "
            "about reconciliation."
        ),
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("concept/finger", "claim_block.concept.finger"),
    )
    result = ingest(
        ws,
        src,
        source_id="src.alpha",
        source_class="markdown_prose",
        seed_claims=[seed],
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)
    return ws


def _lint_warnings(ws: Workspace) -> list[str]:
    return list(lint(ws).warnings)


def _has_stale_fingerprint_warning(ws: Workspace, page_id: str) -> bool:
    return any(
        f"stale render fingerprint for page_id={page_id}" == w
        for w in _lint_warnings(ws)
    )


def test_render_clears_stale_fingerprint_warning(tmp_path: Path) -> None:
    """A subsequent ``render(...)`` must clear a stale-fingerprint
    warning the previous lint reported.
    """
    ws = _seed_one_entity_workspace(tmp_path)
    fps = FingerprintStore(ws)

    # Corrupt the stored fingerprint so lint flags it as stale.
    fps.set("concept/finger", "sha256:deadbeef")
    assert _has_stale_fingerprint_warning(ws, "concept/finger"), (
        "precondition: corrupted fingerprint must surface as a lint warning"
    )

    # A successful render should rewrite the fingerprint.
    result = render(ws)
    affected = result.rendered_pages + result.unchanged_pages
    assert any(p.endswith("finger.md") for p in affected), affected

    # Fresh lint is clean for that page id.
    assert not _has_stale_fingerprint_warning(ws, "concept/finger"), (
        f"stale-fingerprint warning survived a successful render: "
        f"{_lint_warnings(ws)}"
    )


def test_reconcile_clears_stale_fingerprint_warning(tmp_path: Path) -> None:
    """``reconcile`` must repair pages whose stored fingerprint
    diverges from the union recomputation, and a fresh lint must
    be clean for those pages afterwards.
    """
    ws = _seed_one_entity_workspace(tmp_path)
    fps = FingerprintStore(ws)
    fps.set("concept/finger", "sha256:deadbeef")
    assert _has_stale_fingerprint_warning(ws, "concept/finger")

    result = reconcile(ws)
    # The page should appear in rerendered_pages because its
    # fingerprint diverged from the recomputed value.
    assert any(p.endswith("finger.md") for p in result.pages_rerendered), (
        f"reconcile did not rerender the divergent page: "
        f"{result.pages_rerendered}"
    )

    assert not _has_stale_fingerprint_warning(ws, "concept/finger"), (
        f"stale-fingerprint warning survived reconcile: "
        f"{_lint_warnings(ws)}"
    )


def test_unchanged_content_stays_clean_after_render(tmp_path: Path) -> None:
    """A second consecutive ``render(...)`` on an up-to-date page
    must report the page as unchanged and leave lint clean.
    """
    ws = _seed_one_entity_workspace(tmp_path)
    # Initial ingest already rendered the page. A second render is
    # the no-op case.
    result = render(ws)
    assert any(p.endswith("finger.md") for p in result.unchanged_pages), (
        f"second render of an unchanged page should land in "
        f"unchanged_pages; got rendered={result.rendered_pages} "
        f"unchanged={result.unchanged_pages}"
    )
    # No stale-fingerprint warning anywhere.
    warnings = _lint_warnings(ws)
    assert not any(
        w.startswith("stale render fingerprint")
        for w in warnings
    ), warnings


def test_pre_write_failure_leaves_prior_fingerprint_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the page-write step fails before completion, the stored
    fingerprint must remain at the prior known-good value.

    Slice 074 moved the mutating render onto the
    ``RenderTransaction`` stage-then-commit primitive. The "pre-write
    failure" surface is now the commit step; this test monkeypatches
    ``RenderTransaction.commit`` to raise after staging completes
    but before any final page/fingerprint replacement, then asserts
    the stored fingerprint is byte-identical to before. The contract
    is unchanged: a failed write must not advance the fingerprint.
    """
    ws = _seed_one_entity_workspace(tmp_path)
    fps = FingerprintStore(ws)
    prior = fps.get("concept/finger")
    assert prior is not None and prior.startswith("sha256:"), prior

    # Restore the page to its pre-render placeholder content so the
    # next render attempt produces a non-trivial diff and the staging
    # path actually fires.
    page_path = ws.pages / "concepts" / "finger.md"
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")

    class _SimulatedWriteFailure(RuntimeError):
        pass

    from llloom.state import render_transactions as txn_mod

    def _failing_commit(self):  # noqa: ARG001
        raise _SimulatedWriteFailure("simulated render commit failure")

    monkeypatch.setattr(txn_mod.RenderTransaction, "commit", _failing_commit)

    with pytest.raises(_SimulatedWriteFailure):
        render(ws)

    # Prior fingerprint must be preserved byte-for-byte.
    after = FingerprintStore(ws).get("concept/finger")
    assert after == prior, (
        f"failed pre-write must not advance the stored fingerprint: "
        f"prior={prior!r}, after={after!r}"
    )


def test_hash_derivation_agrees_across_render_and_lint(tmp_path: Path) -> None:
    """The fingerprint ``render`` writes for a page must equal the
    value ``compute_page_render_fingerprints`` recomputes for the
    same page — including the multi-entity union case.
    """
    ws = _seed_one_entity_workspace(tmp_path)
    # Add a second entity targeting the same page+block.
    src_b = ws.raw_sources / "beta.md"
    src_b.write_text(SOURCE_TEXT_BETA, encoding="utf-8")
    beta_seed = SeedClaim(
        entity_id="concept.beta",
        entity_type="concept",
        display_name="Beta",
        claim_id="c.beta.1",
        claim_kind="definition",
        claim_text=(
            "The beta entity contributes a second claim to the same "
            "shared block."
        ),
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("concept/finger", "claim_block.concept.finger"),
    )
    result = ingest(
        ws,
        src_b,
        source_id="src.beta",
        source_class="markdown_prose",
        seed_claims=[beta_seed],
    )
    assert result.succeeded, (result.refusal_reason, result.extraction_errors)

    # After the union ingest the stored fingerprint must equal the
    # recomputed union fingerprint.
    fps = FingerprintStore(ws)
    stored = fps.get("concept/finger")
    assert stored is not None
    store = ClaimStore(ws)
    recomputed = compute_page_render_fingerprints(store.iter_entities())
    assert recomputed["concept/finger"] == stored, (
        f"render-side fingerprint ({stored!r}) disagrees with "
        f"lint/health-side recomputation "
        f"({recomputed['concept/finger']!r}); the symptom that "
        f"caused field-report warnings to survive successful renders"
    )

    # Lint must be clean for the page.
    warnings = _lint_warnings(ws)
    assert not any(
        "stale render fingerprint for page_id=concept/finger" in w
        for w in warnings
    ), warnings
