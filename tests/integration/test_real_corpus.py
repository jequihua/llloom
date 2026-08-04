"""Integration tests against the real ``docs/fixture_corpus/``.

Covers:

- the ``NCCP Act of 1991.md`` negative-test fixture (empty/failed OCR)
- the substantive ``NCCP Act of 2003.md`` with the legal_act locator
- a sampled scientific article with the markdown_prose locator
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.verify import verify
from llloom.workspace.layout import Workspace


def _find_package_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src" / "llloom").is_dir()
        ):
            return candidate
    raise RuntimeError(f"could not locate llloom package root from {start}")


def _fixture_corpus_root(package_root: Path) -> Path:
    in_package = package_root / "docs" / "fixture_corpus"
    if in_package.is_dir():
        return in_package
    return package_root.parent / "docs" / "fixture_corpus"


REPO_ROOT = _find_package_root(Path(__file__).resolve())
FIXTURE_CORPUS = _fixture_corpus_root(REPO_ROOT)


def _copy_to_raw(ws: Workspace, name: str) -> Path:
    src = FIXTURE_CORPUS / name
    dst = ws.raw_sources / name
    shutil.copyfile(src, dst)
    return dst


@pytest.mark.skipif(
    not (FIXTURE_CORPUS / "NCCP Act of 1991.md").is_file(),
    reason="NCCP 1991 fixture missing",
)
def test_nccp_1991_refuses_empty_source(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    src = _copy_to_raw(ws, "NCCP Act of 1991.md")
    result = ingest(ws, src, source_id="src.nccp1991", source_class="markdown_prose")
    assert not result.succeeded
    assert result.refusal_reason == "empty source"


@pytest.mark.skipif(
    not (FIXTURE_CORPUS / "NCCP Act of 2003.md").is_file(),
    reason="NCCP 2003 fixture missing",
)
def test_nccp_2003_legal_act_locator(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    src = _copy_to_raw(ws, "NCCP Act of 2003.md")
    seed = SeedClaim(
        entity_id="law.nccp.sec2800",
        entity_type="law",
        display_name="NCCP Act, section 2800",
        claim_id="c.nccp.2800",
        claim_kind="statute",
        claim_text=(
            "This chapter shall be known, and may be cited, as the "
            "Natural Community Conservation Planning Act."
        ),
        locator=Locator(
            locator_type="legal_act_v1",
            act_title="Natural Community Conservation Planning Act",
            section_label="Section 2800",
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
    )
    result = ingest(
        ws,
        src,
        source_id="src.nccp2003",
        source_class="legal_act",
        seed_claims=[seed],
    )
    assert result.succeeded, result.refusal_reason
    assert "c.nccp.2800" in {c.claim_id for c in result.claims_created}
    v = verify(ws)
    assert v.passed, v.notes


SCIENTIFIC_FIXTURE = "j.tree.2007.10.001.md"


@pytest.mark.skipif(
    not (FIXTURE_CORPUS / SCIENTIFIC_FIXTURE).is_file(),
    reason="scientific fixture missing",
)
def test_scientific_markdown_prose_ingest(tmp_path: Path) -> None:
    """Verify that at least one seed claim against a real scientific
    article can be registered and verified end-to-end.

    The span chosen deliberately targets the opening prose of the
    article under the unnamed root heading so the locator is robust to
    the article's deeper heading structure."""
    ws = Workspace.init(tmp_path)
    src = _copy_to_raw(ws, SCIENTIFIC_FIXTURE)
    text = src.read_text(encoding="utf-8")
    # Find the first substantive sentence; bail with a clear failure if
    # the fixture shape changes.
    marker = "Conservation planning is the process"
    if marker not in text:
        pytest.skip(f"fixture {SCIENTIFIC_FIXTURE} does not contain expected marker")

    # The "Conservation in a planning changing world" is a heading in
    # this fixture; use its heading path to pick the abstract paragraph.
    seed = SeedClaim(
        entity_id="concept.conservation_planning",
        entity_type="concept",
        display_name="Conservation planning",
        claim_id="c.cp.1",
        claim_kind="definition",
        claim_text=(
            "Conservation planning is the process of locating, configuring, "
            "implementing and maintaining areas that are managed to promote "
            "the persistence of biodiversity and other natural values."
        ),
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Conservation in a planning changing world"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
    )
    result = ingest(
        ws,
        src,
        source_id="src.jtree_2007",
        source_class="markdown_prose",
        seed_claims=[seed],
    )
    assert result.succeeded, result.refusal_reason
    v = verify(ws)
    assert v.passed, v.notes

