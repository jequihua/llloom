"""M003/S01 unit tests: the explicit opt-in profiled page-create request.

Covers the library request surface, the deterministic emission contract,
and the rollback contract. Every expected byte oracle here is an
independent literal built in this module; none is produced by calling
the production renderer under test.
"""

from __future__ import annotations

import hashlib

import pytest

from llloom.ops.page import CANDIDATE_FRAMEWORK_PROFILE, PageCreateError, create_page
from llloom.okf import observe_page_okf
from llloom.pages.regions import parse_page
from llloom.workspace.layout import Workspace

CRLF = b"\r\n"


# ---- independent oracles ----------------------------------------------------
#
# Written out by hand from the accepted emission contract in Coding Prompt 014.
# Deliberately NOT derived from `_render_stub`.


def canonical_legacy_lf(page_id: str, page_class: str, title: str, marker_tail: str) -> bytes:
    """The canonical LF legacy rendering. Independent literal oracle."""
    return (
        "---\n"
        f"page_id: {page_id}\n"
        f"page_class: {page_class}\n"
        "write_policy: mixed\n"
        "status: draft\n"
        "---\n"
        "\n"
        f"# {title}\n"
        "\n"
        f"<!-- llloom:claim-block id=claim_block.{marker_tail} -->\n"
        "\n"
        "<!-- /llloom:claim-block -->\n"
        "\n"
        f"<!-- llloom:commentary id=commentary.{marker_tail} owner=human -->\n"
        "\n"
        "<!-- /llloom:commentary -->\n"
    ).encode("utf-8")


def canonical_profiled_lf(page_id: str, page_class: str, title: str, marker_tail: str) -> bytes:
    """The canonical LF profiled rendering. Independent literal oracle."""
    return (
        "---\n"
        "type: page\n"
        'framework_profile: "0.1-rc.1"\n'
        f"page_id: {page_id}\n"
        f"page_class: {page_class}\n"
        "write_policy: mixed\n"
        "status: draft\n"
        "---\n"
        "\n"
        f"# {title}\n"
        "\n"
        f"<!-- llloom:claim-block id=claim_block.{marker_tail} -->\n"
        "\n"
        "<!-- /llloom:claim-block -->\n"
        "\n"
        f"<!-- llloom:commentary id=commentary.{marker_tail} owner=human -->\n"
        "\n"
        "<!-- /llloom:commentary -->\n"
    ).encode("utf-8")


#: The exact removable compatibility block, LF form.
PROFILE_BLOCK = b'type: page\nframework_profile: "0.1-rc.1"\n'


def workspace_fingerprint(ws: Workspace) -> set[str]:
    return {p.relative_to(ws.root).as_posix() for p in ws.root.rglob("*")}


# ---- request surface --------------------------------------------------------


def test_the_accepted_value_is_the_single_pinned_candidate():
    assert CANDIDATE_FRAMEWORK_PROFILE == "0.1-rc.1"


def test_omitting_the_opt_in_is_the_legacy_path(fresh_workspace):
    ws = fresh_workspace
    result = create_page(ws, page_id="concept/legacy-demo", title="Legacy Demo")
    raw = (ws.root / result.page_path).read_bytes()
    # Legacy keeps the accepted native writer behavior, so the physical
    # representation is host-dependent. Normalize before comparing content.
    assert raw.replace(CRLF, b"\n") == canonical_legacy_lf(
        "concept/legacy-demo", "concept", "Legacy Demo", "concept.legacy-demo"
    )
    assert b"framework_profile" not in raw
    assert b"type: page" not in raw


@pytest.mark.parametrize(
    "value",
    ["", "0.1", "0.1-rc.2", "0.2-rc.1", "1.0", " 0.1-rc.1", "0.1-rc.1 ", "None"],
)
def test_unsupported_library_values_refuse_before_any_operation(fresh_workspace, value):
    ws = fresh_workspace
    before = workspace_fingerprint(ws)
    with pytest.raises(PageCreateError) as excinfo:
        create_page(ws, page_id="concept/nope", framework_profile=value)
    assert "unsupported framework_profile" in str(excinfo.value)
    # Nothing was created: no page, no directory, no journal entry, no lock.
    assert workspace_fingerprint(ws) == before
    assert not (ws.pages / "concepts" / "nope.md").exists()


def test_refusal_precedes_page_id_validation_so_no_operation_opens(fresh_workspace):
    """The profile check runs first, so even a malformed id refuses on profile."""
    ws = fresh_workspace
    before = workspace_fingerprint(ws)
    with pytest.raises(PageCreateError) as excinfo:
        create_page(ws, page_id="../escape", framework_profile="bogus")
    assert "unsupported framework_profile" in str(excinfo.value)
    assert workspace_fingerprint(ws) == before


def test_the_opt_in_is_request_local(fresh_workspace):
    """A profiled call cannot make a later omitted call profiled."""
    ws = fresh_workspace
    first = create_page(
        ws, page_id="concept/first", title="First", framework_profile="0.1-rc.1"
    )
    second = create_page(ws, page_id="concept/second", title="Second")
    assert b"framework_profile" in (ws.root / first.page_path).read_bytes()
    assert b"framework_profile" not in (ws.root / second.page_path).read_bytes()


def test_result_dataclass_shape_is_unchanged_on_both_paths(fresh_workspace):
    ws = fresh_workspace
    legacy = create_page(ws, page_id="concept/a", title="A")
    profiled = create_page(
        ws, page_id="concept/b", title="B", framework_profile="0.1-rc.1"
    )
    expected = {
        "page_id",
        "page_class",
        "page_path",
        "claim_block_id",
        "commentary_id",
        "status",
        "op_id",
        "refusal_reason",
    }
    assert set(legacy.__dataclass_fields__) == expected
    assert set(profiled.__dataclass_fields__) == expected
    for field in ("framework_profile", "profile_valid", "profile"):
        assert not hasattr(profiled, field)


# ---- deterministic profiled bytes -------------------------------------------


def test_profiled_bytes_match_the_independent_literal_oracle(fresh_workspace):
    ws = fresh_workspace
    result = create_page(
        ws, page_id="concept/handoff", title="Handoff", framework_profile="0.1-rc.1"
    )
    raw = (ws.root / result.page_path).read_bytes()
    assert raw == canonical_profiled_lf(
        "concept/handoff", "concept", "Handoff", "concept.handoff"
    )


def test_profiled_creation_emits_lf_bytes_and_one_final_lf(fresh_workspace):
    ws = fresh_workspace
    result = create_page(
        ws, page_id="concept/lf", title="Lf", framework_profile="0.1-rc.1"
    )
    raw = (ws.root / result.page_path).read_bytes()
    assert CRLF not in raw
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")


def test_the_two_added_fields_are_first_and_in_contract_order(fresh_workspace):
    ws = fresh_workspace
    result = create_page(
        ws, page_id="concept/order", title="Order", framework_profile="0.1-rc.1"
    )
    lines = (ws.root / result.page_path).read_bytes().split(b"\n")
    assert lines[0] == b"---"
    assert lines[1] == b"type: page"
    assert lines[2] == b'framework_profile: "0.1-rc.1"'
    assert lines[3] == b"page_id: concept/order"
    assert lines[4] == b"page_class: concept"
    assert lines[5] == b"write_policy: mixed"
    assert lines[6] == b"status: draft"
    assert lines[7] == b"---"


def test_no_unearned_field_is_emitted(fresh_workspace):
    ws = fresh_workspace
    result = create_page(
        ws, page_id="concept/lean", title="Lean", framework_profile="0.1-rc.1"
    )
    text = (ws.root / result.page_path).read_text(encoding="utf-8")
    frontmatter = text.split("---\n")[1]
    for forbidden in (
        "framework_id",
        "llloom:",
        "generated_by",
        "created_at",
        "schema_version",
        "profile_version",
        "timestamp",
    ):
        assert forbidden not in frontmatter, forbidden
    # `framework_id` must not appear anywhere in the page, not just frontmatter.
    assert "framework_id" not in text


@pytest.mark.parametrize(
    "page_id,page_class,tail",
    [
        ("concept/inferred", None, "concept.inferred"),
        ("bare-page", "concept", "bare-page"),
        ("entity/thing", None, "entity.thing"),
        ("navigation/nav", None, "navigation.nav"),
    ],
)
def test_bare_and_inferred_class_forms_are_deterministic(
    tmp_path, page_id, page_class, tail
):
    """Two clean workspaces produce identical bytes for the same request."""
    digests = []
    for index in (0, 1):
        ws = Workspace.init(tmp_path / f"ws{index}")
        result = create_page(
            ws,
            page_id=page_id,
            page_class=page_class,
            title="T",
            framework_profile="0.1-rc.1",
        )
        raw = (ws.root / result.page_path).read_bytes()
        digests.append(hashlib.sha256(raw).hexdigest())
        assert f"claim_block.{tail}".encode() in raw
    assert digests[0] == digests[1]


# ---- rollback ---------------------------------------------------------------


def test_removing_the_profile_block_recovers_the_canonical_lf_legacy_rendering(
    fresh_workspace,
):
    """Required test 7, measured against the independent LF oracle."""
    ws = fresh_workspace
    profiled = create_page(
        ws, page_id="concept/roll", title="Roll", framework_profile="0.1-rc.1"
    )
    raw = (ws.root / profiled.page_path).read_bytes()
    assert raw.count(PROFILE_BLOCK) == 1
    rolled_back = raw.replace(PROFILE_BLOCK, b"", 1)
    assert rolled_back == canonical_legacy_lf(
        "concept/roll", "concept", "Roll", "concept.roll"
    )


def test_windows_legacy_normalized_to_lf_equals_the_canonical_lf_oracle(
    fresh_workspace,
):
    """The physical legacy page may be CRLF; its LF normalization is the oracle."""
    ws = fresh_workspace
    legacy = create_page(ws, page_id="concept/roll", title="Roll")
    physical = (ws.root / legacy.page_path).read_bytes()
    oracle = canonical_legacy_lf("concept/roll", "concept", "Roll", "concept.roll")
    assert physical.replace(CRLF, b"\n") == oracle
    # And the representation is exactly one of the two accepted forms.
    assert physical in (oracle, oracle.replace(b"\n", CRLF))


def test_rolled_back_profiled_page_equals_normalized_physical_legacy(fresh_workspace):
    """End-to-end rollback identity across both writers."""
    ws = fresh_workspace
    profiled = create_page(
        ws, page_id="concept/same", title="Same", framework_profile="0.1-rc.1"
    )
    ws2 = Workspace.init(ws.root.parent / "legacy-ws")
    legacy = create_page(ws2, page_id="concept/same", title="Same")
    rolled = (ws.root / profiled.page_path).read_bytes().replace(PROFILE_BLOCK, b"", 1)
    assert rolled == (ws2.root / legacy.page_path).read_bytes().replace(CRLF, b"\n")


# ---- semantics --------------------------------------------------------------


def test_created_profiled_page_parses_natively_and_observes_as_exact_r14(
    fresh_workspace,
):
    ws = fresh_workspace
    result = create_page(
        ws, page_id="concept/r14", title="R14", framework_profile="0.1-rc.1"
    )
    page_path = ws.root / result.page_path

    parsed = parse_page(page_path.read_text(encoding="utf-8"))
    assert parsed.frontmatter["page_id"] == "concept/r14"
    assert parsed.frontmatter["page_class"] == "concept"
    assert parsed.frontmatter["write_policy"] == "mixed"
    assert parsed.frontmatter["status"] == "draft"
    assert parsed.frontmatter["type"] == "page"
    assert parsed.frontmatter["framework_profile"] == "0.1-rc.1"
    assert parsed.claim_block_id == "claim_block.concept.r14"
    assert parsed.commentary_id == "commentary.concept.r14"

    observation = observe_page_okf(page_path)
    assert observation.read_status == "ok"
    assert observation.okf_concept_result == "pass"
    assert observation.okf_concept_reason is None
    assert observation.framework_profile_result == "pass"
    assert observation.framework_profile_reason is None
    assert observation.declared_framework_profile == "0.1-rc.1"
    assert observation.execution_eligibility == "not_evaluated"


def test_legacy_page_still_observes_as_r9_and_is_not_a_repair_target(fresh_workspace):
    """Profile validity grants nothing, and ordinary native pages stay ordinary."""
    ws = fresh_workspace
    result = create_page(ws, page_id="concept/plain", title="Plain")
    observation = observe_page_okf(ws.root / result.page_path)
    assert observation.okf_concept_result == "fail"
    assert observation.okf_concept_reason == "OKF_TYPE_MISSING"
    assert observation.framework_profile_result == "not_applicable"
    assert observation.execution_eligibility == "not_evaluated"


def test_profile_validity_grants_no_execution_eligibility(fresh_workspace):
    ws = fresh_workspace
    result = create_page(
        ws, page_id="concept/auth", title="Auth", framework_profile="0.1-rc.1"
    )
    observation = observe_page_okf(ws.root / result.page_path)
    assert observation.framework_profile_result == "pass"
    assert observation.execution_eligibility == "not_evaluated"
