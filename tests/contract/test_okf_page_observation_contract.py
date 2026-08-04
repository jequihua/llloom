"""Contract tests for the opt-in OKF observation surface.

These assert the frozen public shape, the read-only and non-authoritative
obligations accepted in M001 section 8, and the P01-P07 compatibility cases from
section 5.4. Expected inventories are written literally, never derived from the
code under test.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import socket
from pathlib import Path

import pytest
import yaml

import llloom
from llloom.okf import PageOkfObservation, observe_page_okf
from llloom.okf import observation as okf_observation
from llloom.ops.doctor import doctor
from llloom.ops.ingest import ingest
from llloom.ops.inspect import list_pages, list_render_targets, list_sources
from llloom.ops.lint import lint
from llloom.ops.page import create_page
from llloom.ops.query import query
from llloom.ops.render import render
from llloom.ops.results import PageSummary, RenderTargetListEntry
from llloom.ops.verify import verify
from llloom.workspace.layout import Workspace

REPO_ROOT = Path(__file__).resolve().parents[2]

PROFILED = 'type: analysis\nframework_profile: "0.1-rc.1"'
MARKERS = (
    "\n# T\n\n"
    "<!-- llloom:claim-block id=cb.t -->\n\n<!-- /llloom:claim-block -->\n\n"
    "<!-- llloom:commentary id=cm.t owner=human -->\n\n<!-- /llloom:commentary -->\n"
)

# The M000 native page-create golden, recorded in
# 03_experiments/m000_llloom_native_baseline_evidence.md. The evidence digest is
# the CRLF byte form the native writer produces on Windows; the LF form is the
# same text on a platform without newline translation.
M000_PAGE_CREATE_CRLF = "7cbd75e8e7b31ab829b5e5a719241a15657d2cbb0d3afa4206f887783c243a2e"
M000_PAGE_CREATE_LF = "11cc2430a7644de01db66930e423f61c2093299e4d1cc12cbb73c8acb2d80ad6"

EXPECTED_VERBS = {
    "claim-card", "doctor", "ingest", "init", "lint", "list-claims", "list-pages",
    "list-render-targets", "list-sources", "list_merge_proposals", "merge-alias",
    "page", "prepare-pdf", "promote", "query", "rebuild", "reconcile", "reject-alias",
    "render", "retract", "review-alias", "seed", "status", "supersede", "unlock",
    "verify",
}
EXPECTED_MCP_TOOLS = {
    "llloom_status", "llloom_query", "llloom_verify", "llloom_lint",
    "llloom_graph_neighbors", "llloom_list_merge_proposals",
}
EXPECTED_PAGE_SUMMARY_FIELDS = (
    "page_id", "page_path", "page_class", "status", "write_policy",
)
EXPECTED_RENDER_TARGET_FIELDS = (
    "page_id", "block_id", "page_path", "marker_health", "marker_message",
    "contributing_claim_ids",
)
EXPECTED_OBSERVATION_FIELDS = (
    "read_status", "okf_concept_result", "okf_concept_reason",
    "framework_profile_result", "framework_profile_reason", "execution_eligibility",
    "declared_framework_profile",
)


def page(body: str, markers: str = MARKERS) -> str:
    return "---\n" + body + "\n---\n" + markers


def tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Every regular file under ``root`` with its bytes and mtime_ns."""
    out: dict[str, tuple[bytes, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = (
                path.read_bytes(), path.stat().st_mtime_ns
            )
    return out


# ---- public surface ----------------------------------------------------------


def test_observation_exposes_exactly_the_accepted_fields_in_order():
    fields = tuple(f.name for f in dataclasses.fields(PageOkfObservation))
    assert fields == EXPECTED_OBSERVATION_FIELDS


def test_observation_dataclass_is_frozen():
    assert dataclasses.fields(PageOkfObservation)
    params = getattr(PageOkfObservation, "__dataclass_params__")
    assert params.frozen is True


def test_okf_package_exports_exactly_two_public_names():
    import llloom.okf as package

    assert package.__all__ == ["PageOkfObservation", "observe_page_okf"]
    public = {name for name in vars(package) if not name.startswith("_")}
    assert public - {"observation", "llloom"} == {
        "PageOkfObservation", "observe_page_okf",
    }


def test_root_export_inventory_is_unchanged():
    assert llloom.__all__ == ["Workspace"]
    assert not hasattr(llloom, "okf") or "okf" not in llloom.__all__


def test_no_framework_id_or_summary_boolean_is_exposed():
    names = {f.name for f in dataclasses.fields(PageOkfObservation)}
    assert "framework_id" not in names
    assert not any(n.startswith("is_") or n.startswith("has_") for n in names)
    assert "marker_health" not in names
    assert not any(
        isinstance(f.type, str) and f.type.strip() == "bool"
        for f in dataclasses.fields(PageOkfObservation)
    )


# ---- P06 public inventory ----------------------------------------------------


def test_p06_public_inventory_unchanged():
    import ast

    tree = ast.parse((REPO_ROOT / "src/llloom/cli.py").read_text(encoding="utf-8"))
    verbs = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "add_parser"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "sub"
        and node.args and isinstance(node.args[0], ast.Constant)
    }
    assert verbs == EXPECTED_VERBS
    assert len(verbs) == 26

    from llloom.mcp_server.tools import TOOL_NAMES

    assert set(TOOL_NAMES) == EXPECTED_MCP_TOOLS

    import llloom.ops.results as results

    declared = [
        name for name in dir(results)
        if isinstance(getattr(results, name), type)
        and dataclasses.is_dataclass(getattr(results, name))
        and getattr(results, name).__module__ == "llloom.ops.results"
    ]
    assert len(declared) == 34


def test_p04_existing_result_shapes_unchanged():
    assert tuple(f.name for f in dataclasses.fields(PageSummary)) == \
        EXPECTED_PAGE_SUMMARY_FIELDS
    assert tuple(f.name for f in dataclasses.fields(RenderTargetListEntry)) == \
        EXPECTED_RENDER_TARGET_FIELDS


# ---- PyYAML boundary ---------------------------------------------------------


def test_module_uses_pure_python_safeloader_only():
    source = inspect.getsource(okf_observation)
    assert "yaml.SafeLoader" in source
    assert "CSafeLoader" not in source
    assert "yaml.Loader" not in source
    assert "yaml.UnsafeLoader" not in source
    assert "yaml.load(" not in source


def test_safeloader_is_used_even_when_csafeloader_is_available(tmp_path, monkeypatch):
    if not hasattr(yaml, "CSafeLoader"):
        pytest.skip("libyaml bindings are not installed in this environment")

    class _Forbidden:
        def __init__(self, *args, **kwargs):
            raise AssertionError("the observation boundary selected a C loader")

    monkeypatch.setattr(yaml, "CSafeLoader", _Forbidden)
    target = tmp_path / "p.md"
    target.write_text(page(PROFILED), encoding="utf-8")
    result = observe_page_okf(target)
    assert result.framework_profile_result == "pass"


def test_declared_pyyaml_range_is_the_candidate_range():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"PyYAML>=6.0.3,<7"' in text


# ---- read-only obligations ---------------------------------------------------


def test_observation_mutates_no_workspace_byte_and_no_mtime(fresh_workspace):
    ws = fresh_workspace
    create_page(ws, page_id="concept/demo", title="Demo")
    profiled = ws.pages / "concepts" / "profiled.md"
    profiled.write_bytes(page(PROFILED).encode("utf-8"))

    before = tree_snapshot(ws.root)
    for target in sorted(ws.pages.rglob("*.md")):
        observe_page_okf(target)
    observe_page_okf(ws.pages / "does_not_exist.md")
    after = tree_snapshot(ws.root)

    assert before == after


def test_observation_creates_no_lock_journal_transaction_sidecar_or_report(fresh_workspace):
    ws = fresh_workspace
    target = ws.pages / "p.md"
    target.write_bytes(page(PROFILED).encode("utf-8"))

    def state_counts() -> dict[str, int]:
        return {
            name: len(list(directory.rglob("*")))
            for name, directory in (
                ("locks", ws.state_locks), ("journals", ws.state_journals),
                ("transactions", ws.state_transactions),
                ("reports_health", ws.state_reports_health),
                ("reports_updates", ws.state_reports_updates),
            )
        }

    before = state_counts()
    observe_page_okf(target)
    assert state_counts() == before
    assert not ws.state_search.exists()
    assert not ws.state_graph.exists()
    assert not ws.state_structure.exists()


def test_observation_opens_no_socket(tmp_path, monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("the observation boundary attempted network access")

    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    target = tmp_path / "p.md"
    target.write_text(page(PROFILED), encoding="utf-8")
    assert observe_page_okf(target).okf_concept_result == "pass"


def test_observation_never_imports_the_reference_script():
    source = inspect.getsource(okf_observation)
    assert "okf_yaml_profile" not in source
    assert "artifact_integrity_preflight" not in source
    assert "scripts" not in source


def test_observation_never_calls_the_native_region_parser():
    """Marker parsing stays independent: no import of, and no call into, the native
    region module. The module docstring may name it; the code may not use it."""
    import ast

    tree = ast.parse(inspect.getsource(okf_observation))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(name.startswith("llloom.pages") for name in imported)
    assert not any("okf_yaml_profile" in name for name in imported)

    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "parse_page" not in called
    assert "FRONTMATTER_RE" not in called


# ---- authority separation, proved by mutation --------------------------------


def _authority_snapshot(ws: Workspace) -> dict:
    """Every authority-bearing and read-only surface M001 section 8 names."""
    return {
        "sources": [dataclasses.astuple(s) for s in list_sources(ws)],
        "pages": [dataclasses.astuple(p) for p in list_pages(ws)],
        "render_targets": [dataclasses.astuple(r) for r in list_render_targets(ws)],
        "verify": dataclasses.astuple(verify(ws)),
        "render_plan": [
            dataclasses.astuple(entry) for entry in render(ws, list_targets=True).plan
        ],
        "query": dataclasses.astuple(query(ws, question="evidence")),
        "lint": dataclasses.astuple(lint(ws)),
        "doctor_ids": sorted(w.warning_id for w in doctor(ws).warnings),
    }


def test_profile_result_change_alters_no_authority_or_default_output(fresh_workspace):
    """Mutation-style probe: flip a page from R14 to R13 and prove nothing moves."""
    ws = fresh_workspace
    create_page(ws, page_id="concept/demo", title="Demo")
    source = ws.raw_sources / "evidence.md"
    source.write_text("# Evidence\n\nA sentence of raw evidence.\n", encoding="utf-8")
    ingest(ws, source, source_class="raw_evidence")

    profiled = ws.pages / "concepts" / "profiled.md"
    profiled.write_bytes(page(PROFILED).encode("utf-8"))
    assert observe_page_okf(profiled).framework_profile_result == "pass"
    before = _authority_snapshot(ws)

    # The only change is the declared profile version: R14 -> R13.
    profiled.write_bytes(
        page('type: analysis\nframework_profile: "9.9-rc.9"').encode("utf-8")
    )
    after_observation = observe_page_okf(profiled)
    assert after_observation.framework_profile_result == "fail"
    assert after_observation.framework_profile_reason == "PROFILE_VERSION_UNSUPPORTED"

    assert _authority_snapshot(ws) == before


def test_execution_eligibility_is_constant_across_every_row(tmp_path):
    bodies = [
        "# legacy\n",
        "---\ntype: analysis\n",
        page(PROFILED),
        page("type: analysis\ntype: other"),
        page('type: "Made Up"'),
        page('type: analysis\nframework_profile: "9.9-rc.9"'),
        page('type: analysis\ntags: ["a"]'),
    ]
    for index, body in enumerate(bodies):
        target = tmp_path / f"c{index}.md"
        target.write_text(body, encoding="utf-8")
        assert observe_page_okf(target).execution_eligibility == "not_evaluated"
    assert observe_page_okf(tmp_path / "absent.md").execution_eligibility == \
        "not_evaluated"


# ---- P01-P03, P05, P07 default-output compatibility ---------------------------


def test_p01_p02_p03_default_outputs_are_unchanged_by_observing(fresh_workspace):
    ws = fresh_workspace
    create_page(ws, page_id="concept/demo", title="Demo")
    (ws.pages / "concepts" / "profiled.md").write_bytes(page(PROFILED).encode("utf-8"))
    # A legacy page: no frontmatter, but a well-formed marker arrangement, so the
    # only thing distinguishing it from the profiled page is profile status.
    (ws.pages / "concepts" / "legacy.md").write_bytes(
        ("# Legacy\n" + MARKERS).encode("utf-8")
    )

    before_pages = [dataclasses.astuple(p) for p in list_pages(ws)]
    before_lint = dataclasses.astuple(lint(ws))
    before_doctor = sorted(w.warning_id for w in doctor(ws).warnings)

    for target in sorted(ws.pages.rglob("*.md")):
        observe_page_okf(target)

    assert [dataclasses.astuple(p) for p in list_pages(ws)] == before_pages
    assert dataclasses.astuple(lint(ws)) == before_lint
    assert sorted(w.warning_id for w in doctor(ws).warnings) == before_doctor
    # P02: the default lint surface is clean, and nothing in it is attributable to
    # profile status -- a legacy page is not a failure for lacking a profile block.
    result = lint(ws)
    assert result.failures == []
    profile_terms = ("framework_profile", "okf", "OKF", "profile")
    assert not [
        message for message in list(result.failures) + list(result.warnings)
        if any(term in message for term in profile_terms)
    ]
    assert before_doctor == sorted({"sidecar:graph:missing", "sidecar:search:missing"})


def test_p05_ingest_verify_render_query_cycle_is_unchanged(fresh_workspace):
    ws = fresh_workspace
    source = ws.raw_sources / "evidence.md"
    source.write_text("# Evidence\n\nA sentence of raw evidence.\n", encoding="utf-8")
    ingest(ws, source, source_class="raw_evidence")
    before = (
        dataclasses.astuple(verify(ws)),
        [dataclasses.astuple(e) for e in render(ws, list_targets=True).plan],
        dataclasses.astuple(query(ws, question="evidence")),
    )
    (ws.pages / "concepts" / "profiled.md").write_bytes(page(PROFILED).encode("utf-8"))
    observe_page_okf(ws.pages / "concepts" / "profiled.md")
    after = (
        dataclasses.astuple(verify(ws)),
        [dataclasses.astuple(e) for e in render(ws, list_targets=True).plan],
        dataclasses.astuple(query(ws, question="evidence")),
    )
    assert after == before


def test_p07_legacy_page_create_is_byte_identical_to_the_m000_golden(fresh_workspace):
    ws = fresh_workspace
    result = create_page(ws, page_id="concept/m000-demo", title="M000 Demo")
    created = ws.root / result.page_path
    raw = created.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = M000_PAGE_CREATE_CRLF if b"\r\n" in raw else M000_PAGE_CREATE_LF
    assert digest == expected
    # The created page carries only the native flat keys and no profile block.
    text = raw.decode("utf-8")
    assert "framework_profile" not in text
    assert "type:" not in text.split("---")[1]


def test_llloom_owned_fixture_manifest_is_present_and_declares_every_case():
    manifest_path = REPO_ROOT / "tests/fixtures/okf_pages/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile_candidate"] == "0.1-rc.1"
    assert manifest["parser_lane"] == "full_parser"
    assert tuple(manifest["observation_fields"]) == EXPECTED_OBSERVATION_FIELDS

    case_ids = {case["id"] for case in manifest["cases"]}
    assert case_ids == {f"L{n:02d}" for n in range(1, 20)}

    hostile_ids = {case["id"] for case in manifest["hostile_cases"]}
    expected_hostile = {
        f"H{n:02d}-{side}"
        for n in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11)
        for side in ("max", "plus1")
    } | {"H05-max", "H05-plus1", "H05-nearmax", "H05-order",
         "H12-cycle", "H13-recursion"}
    assert hostile_ids == expected_hostile

    compat_ids = {case["id"] for case in manifest["compatibility_cases"]}
    assert compat_ids == {f"P{n:02d}" for n in range(1, 8)}


# ---- M002 amendment: the R0U public vocabulary ------------------------------

ACCEPTED_READ_STATUS = {"ok", "target_absent", "oversize", "undecodable", "unreadable"}


#: Development-only M001 authority. The M006/S01 projection contract excludes the
#: analysis plane from the front release, so only the document check below depends
#: on it -- never the product assertion.
M001_AMENDMENT = (
    REPO_ROOT / "02_analysis" / "llloom_okf_consumer_contract_and_fixture_plan.md"
)


def test_read_status_vocabulary_matches_the_amendment_and_manifest():
    """Product assertion: the release-contained llloom fixture manifest declares
    exactly the accepted read-status vocabulary."""
    manifest = json.loads(
        (REPO_ROOT / "tests/fixtures/okf_pages/manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["read_status_vocabulary"]) == ACCEPTED_READ_STATUS


@pytest.mark.skipif(
    not M001_AMENDMENT.is_file(),
    reason="development-only M001 authority is excluded from the release projection",
)
def test_the_m001_amendment_document_records_the_r0u_read_status():
    """Development authority: the accepted M001 amendment names R0U/unreadable."""
    amendment = M001_AMENDMENT.read_text(encoding="utf-8")
    assert "M002 Change-Control Amendment" in amendment
    assert "R0U" in amendment
    assert "unreadable" in amendment


def test_the_correction_added_no_public_field():
    assert tuple(f.name for f in dataclasses.fields(PageOkfObservation)) == \
        EXPECTED_OBSERVATION_FIELDS
    assert len(EXPECTED_OBSERVATION_FIELDS) == 7


# ---- authority separation across every named surface ------------------------

SOURCE_TEXT = """# Methods

Complementarity prioritizes sites that add features not already represented in the selected set. It is commonly used to identify gaps.

Second paragraph irrelevant to the claim.
"""

CLAIM_PAGE = """---
page_id: concept/complementarity
page_class: concept
write_policy: mixed
status: draft
---

# Complementarity

<!-- llloom:claim-block id=claim_block.concept.complementarity -->

<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.complementarity owner=human -->

Human commentary that must be preserved.

<!-- /llloom:commentary -->
"""


def _workspace_with_a_real_claim(ws: Workspace):
    """Seed one real claim through the native model-free path.

    The previous version of this probe ingested an `index_only` source, so it had
    no claim whose lifecycle could change -- which is exactly why its authority
    evidence was incomplete.
    """
    from llloom.claims.locators import Locator
    from llloom.ops.ingest import SeedClaim

    source = ws.raw_sources / "article.md"
    source.write_text(SOURCE_TEXT, encoding="utf-8")
    target = ws.pages / "concepts" / "complementarity.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(CLAIM_PAGE, encoding="utf-8")
    seed = SeedClaim(
        entity_id="concept.complementarity",
        entity_type="concept",
        display_name="Complementarity",
        claim_id="c_0001",
        claim_kind="definition",
        claim_text=(
            "Complementarity prioritizes sites that add features not already "
            "represented in the selected set."
        ),
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("concept/complementarity",
                       "claim_block.concept.complementarity"),
    )
    result = ingest(ws, source, source_id="src.article",
                    source_class="markdown_prose", seed_claims=[seed])
    assert result.succeeded, result.refusal_reason
    return result


REQUIRED_MATRIX_NAMES = ("ALLOWED_OPERATIONS", "ALLOWED_READ_CLASSES",
                         "ALLOWED_WRITE_KINDS")

# The exact accepted matrix values, written as independent literals. Sorting a
# dictionary's keys is not a value oracle; these are the values themselves.
EXPECTED_ALLOWED_OPERATIONS = ("ingest", "lint", "query", "render")
EXPECTED_ALLOWED_READ_CLASSES = {
    "ingest": ("ClaimRecord", "SchemaDocument", "SourceDocument",
               "StructureItemContext"),
    "render": ("ClaimBlockRegion", "ClaimRecord", "SchemaDocument"),
    "query": ("ClaimBlockRegion", "ClaimRecord", "SchemaDocument", "SourceSpan"),
    "lint": ("ClaimRecord", "SchemaDocument"),
}
EXPECTED_ALLOWED_WRITE_KINDS = {
    "ingest": ("claim_entity", "journal", "merge_proposal"),
    "render": ("claim_block", "journal", "render_fingerprint"),
    "query": (),
    "lint": ("journal", "report"),
}


def _canonical_matrices(module) -> dict:
    """Canonicalize every nested key and value of all three required matrices.

    Raises ``AttributeError`` if any required name is absent -- which is the point:
    the previous helper filtered missing names out and asserted only that something
    was found, so deleting two of the three still satisfied it.
    """
    missing = [name for name in REQUIRED_MATRIX_NAMES if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"required model matrices absent: {missing}")
    return {
        "ALLOWED_OPERATIONS": tuple(sorted(module.ALLOWED_OPERATIONS)),
        "ALLOWED_READ_CLASSES": {
            key: tuple(sorted(value))
            for key, value in sorted(module.ALLOWED_READ_CLASSES.items())
        },
        "ALLOWED_WRITE_KINDS": {
            key: tuple(sorted(value))
            for key, value in sorted(module.ALLOWED_WRITE_KINDS.items())
        },
    }


def test_all_three_model_matrices_match_their_exact_accepted_values():
    from llloom.llm import harness

    canonical = _canonical_matrices(harness)
    assert canonical["ALLOWED_OPERATIONS"] == EXPECTED_ALLOWED_OPERATIONS
    assert canonical["ALLOWED_READ_CLASSES"] == {
        key: value for key, value in sorted(EXPECTED_ALLOWED_READ_CLASSES.items())
    }
    assert canonical["ALLOWED_WRITE_KINDS"] == {
        key: value for key, value in sorted(EXPECTED_ALLOWED_WRITE_KINDS.items())
    }


@pytest.mark.parametrize("name", REQUIRED_MATRIX_NAMES)
def test_removing_any_required_matrix_fails_the_helper(name, monkeypatch):
    """Mutation strength: 'at least one exists' would pass this; the helper must not."""
    from llloom.llm import harness

    monkeypatch.delattr(harness, name)
    with pytest.raises(AttributeError, match="required model matrices absent"):
        _canonical_matrices(harness)


@pytest.mark.parametrize("name", REQUIRED_MATRIX_NAMES)
def test_altering_any_matrix_value_fails_the_oracle(name, monkeypatch):
    """A changed allow-list value must fail, not merely a missing name."""
    from llloom.llm import harness

    current = getattr(harness, name)
    if isinstance(current, dict):
        altered = dict(current)
        altered["query"] = frozenset({"SomethingNewlyAllowed"})
    else:
        altered = frozenset(set(current) | {"newly_allowed_operation"})
    monkeypatch.setattr(harness, name, altered)
    with pytest.raises(AssertionError):
        test_all_three_model_matrices_match_their_exact_accepted_values()


# ---- meaningful permission execution, allowed and refused --------------------


def test_allowed_in_memory_invocation_records_its_operation_reads_and_writes():
    """One safe allowed path, executed. No filesystem mutation, model call, or
    network: the NullModel backend produces deterministic empty output."""
    from llloom.llm.harness import (
        ClaimRecord, LLMInvoke, NullModel, SchemaDocument, WriteTarget,
    )

    harness = LLMInvoke(model=NullModel())
    output, log = harness.invoke(
        op_id="op.m002.authority",
        operation_kind="lint",
        claim_records=[ClaimRecord(claim_id="c_0001", entity_id="concept.x",
                                   claim_text="A claim under lint.")],
        schema_documents=[SchemaDocument(name="page_classes", text="classes: {}")],
        write_targets=[WriteTarget(path="state/journals/op.m002.authority.yaml",
                                   kind="journal")],
    )
    assert isinstance(output, str)
    assert log.op_id == "op.m002.authority"
    assert log.operation_kind == "lint"
    assert log.refusal is None
    assert sorted(entry["class"] for entry in log.read_inputs) == [
        "ClaimRecord", "SchemaDocument",
    ]
    assert log.model_identifier == "null-model/v0"
    assert [target["kind"] for target in log.write_targets] == ["journal"]


REFUSALS = [
    ("unknown operation", dict(op_id="op.x", operation_kind="not_a_real_op")),
    ("forbidden read class", dict(
        op_id="op.y", operation_kind="lint",
        source_documents="SOURCE_DOCUMENT_PLACEHOLDER")),
    ("forbidden write kind", dict(
        op_id="op.z", operation_kind="query",
        write_targets="WRITE_TARGET_PLACEHOLDER")),
]


@pytest.mark.parametrize("label,kwargs", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_forbidden_permission_paths_are_refused(label, kwargs):
    from llloom.llm.harness import (
        HarnessRefusal, LLMInvoke, NullModel, SourceDocument, WriteTarget,
    )

    call = dict(kwargs)
    if call.get("source_documents") == "SOURCE_DOCUMENT_PLACEHOLDER":
        # `lint` may not read a SourceDocument.
        call["source_documents"] = [SourceDocument(
            source_id="src.x", source_class="markdown_prose", text="body")]
    if call.get("write_targets") == "WRITE_TARGET_PLACEHOLDER":
        # `query` declares no permitted write kind at all.
        call["write_targets"] = [WriteTarget(
            path="claims/entities/concept.x.yaml", kind="claim_entity")]

    harness = LLMInvoke(model=NullModel())
    with pytest.raises(HarnessRefusal):
        harness.invoke(**call)


def _harness_module():
    from llloom.llm import harness

    return harness


def _permission_decisions() -> tuple:
    """Executed permission evidence: one allowed decision and three refusals.

    This replaces the previous `_operation_permission()` tuple, which recorded a
    callable object, two directory booleans, and a lock listing -- and executed no
    permission decision at all. Nothing here touches the filesystem, a model, or the
    network: `NullModel` produces deterministic empty output.
    """
    from llloom.llm.harness import (
        ClaimRecord, HarnessRefusal, LLMInvoke, NullModel, SchemaDocument,
        SourceDocument, WriteTarget,
    )

    harness = LLMInvoke(model=NullModel())
    _, log = harness.invoke(
        op_id="op.m002.snapshot",
        operation_kind="lint",
        claim_records=[ClaimRecord(claim_id="c_0001", entity_id="concept.x",
                                   claim_text="A claim under lint.")],
        schema_documents=[SchemaDocument(name="page_classes", text="classes: {}")],
        write_targets=[WriteTarget(path="state/journals/op.m002.snapshot.yaml",
                                   kind="journal")],
    )
    allowed = (
        log.operation_kind,
        tuple(sorted(entry["class"] for entry in log.read_inputs)),
        tuple(sorted(target["kind"] for target in log.write_targets)),
        log.refusal,
    )

    refusals = []
    for label, call in (
        ("unknown_operation", dict(op_id="op.a", operation_kind="not_a_real_op")),
        ("forbidden_read_class", dict(
            op_id="op.b", operation_kind="lint",
            source_documents=[SourceDocument(source_id="src.x",
                                             source_class="markdown_prose",
                                             text="body")])),
        ("forbidden_write_kind", dict(
            op_id="op.c", operation_kind="query",
            write_targets=[WriteTarget(path="claims/entities/concept.x.yaml",
                                       kind="claim_entity")])),
    ):
        try:
            harness.invoke(**call)
        except HarnessRefusal:
            refusals.append((label, "refused"))
        else:  # pragma: no cover - a permitted forbidden path is the failure
            refusals.append((label, "ALLOWED"))
    return (allowed, tuple(refusals))


def _claim_lifecycle_state(ws: Workspace) -> list[tuple]:
    """Every claim's lifecycle and verification state, not merely a listing."""
    from llloom.claims.store import ClaimStore

    out = []
    for entity in ClaimStore(ws).iter_entities():
        for assertion in entity.assertions:
            out.append((
                entity.entity_id, assertion.claim_id, assertion.status,
                assertion.verification_status, assertion.claim_kind,
                tuple(sorted(ev.source_id for ev in assertion.evidence)),
                tuple(sorted(ev.excerpt_hash or "" for ev in assertion.evidence)),
                tuple(sorted((rt.page_id, rt.block_id)
                             for rt in assertion.render_targets)),
            ))
    return sorted(out)


def _source_registry_state(ws: Workspace) -> list[tuple]:
    from llloom.sources.registry import SourceRegistry

    return sorted(
        (r.source_id, r.source_class, r.status, r.content_hash, r.byte_size)
        for r in SourceRegistry(ws).iter_records()
    )


def _full_authority_snapshot(ws: Workspace) -> dict:
    from llloom.mcp_server.tools import TOOL_NAMES

    return {
        "claims": _claim_lifecycle_state(ws),
        "sources": _source_registry_state(ws),
        "pages": [dataclasses.astuple(p) for p in list_pages(ws)],
        "render_targets": [dataclasses.astuple(r) for r in list_render_targets(ws)],
        "verify": dataclasses.astuple(verify(ws)),
        "render_plan": [
            dataclasses.astuple(e) for e in render(ws, list_targets=True).plan
        ],
        "query": dataclasses.astuple(query(ws, question="complementarity")),
        "lint": dataclasses.astuple(lint(ws)),
        "doctor_ids": sorted(w.warning_id for w in doctor(ws).warnings),
        "llm_matrices": _canonical_matrices(_harness_module()),
        "mcp": tuple(sorted(TOOL_NAMES)),
        "permission_decisions": _permission_decisions(),
    }


PROFILE_MUTATIONS = [
    ("R13 unsupported version", 'type: analysis\nframework_profile: "9.9-rc.9"',
     "fail", "PROFILE_VERSION_UNSUPPORTED"),
    ("R10 out of subset",
     'type: analysis\nframework_profile: "0.1-rc.1"\ntags: ["a"]',
     "fail", "PROFILE_YAML_OUT_OF_SUBSET"),
    ("R11 unknown type", 'type: "Made Up"\nframework_profile: "0.1-rc.1"',
     "fail", "PROFILE_TYPE_UNSUPPORTED"),
    ("R12 no profile field", 'type: analysis\ntitle: "x"',
     "not_applicable", None),
]


@pytest.mark.parametrize("label,body,result,reason", PROFILE_MUTATIONS,
                         ids=[m[0] for m in PROFILE_MUTATIONS])
def test_profile_result_change_alters_no_named_authority_surface(
    fresh_workspace, label, body, result, reason
):
    """Mutation-style probe over every surface M001 section 8 names: a real claim's
    lifecycle and verification status, source-registry status and content hash,
    render eligibility and plan, query inclusion, the model allow-list matrices,
    the MCP tool set, and operation-execution permission."""
    ws = fresh_workspace
    _workspace_with_a_real_claim(ws)

    profiled = ws.pages / "concepts" / "profiled.md"
    profiled.write_bytes(page(PROFILED).encode("utf-8"))
    assert observe_page_okf(profiled).framework_profile_result == "pass"
    before = _full_authority_snapshot(ws)
    assert before["claims"], "the probe must observe at least one real claim"

    profiled.write_bytes(page(body).encode("utf-8"))
    after_observation = observe_page_okf(profiled)
    assert after_observation.framework_profile_result == result, label
    assert after_observation.framework_profile_reason == reason, label

    assert _full_authority_snapshot(ws) == before, label


def test_the_authority_probe_observes_real_lifecycle_values(fresh_workspace):
    """Guards the probe above: a snapshot with no claim proves nothing about claim
    lifecycle, which is what the pre-correction version of this test did."""
    from llloom.claims.models import CLAIM_STATUSES, VERIFICATION_STATUSES

    ws = fresh_workspace
    _workspace_with_a_real_claim(ws)
    claims = _claim_lifecycle_state(ws)
    assert claims, "no claim was seeded"
    for entry in claims:
        assert entry[2] in CLAIM_STATUSES
        assert entry[3] in VERIFICATION_STATUSES
        assert entry[7], "the claim must carry a render target"


def test_observation_cannot_reach_the_model_allow_lists():
    """Source evidence: the observation module names no allow-list constant."""
    source = inspect.getsource(okf_observation)
    for name in ("ALLOWED_OPERATIONS", "ALLOWED_READ_CLASSES", "ALLOWED_WRITE_KINDS",
                 "LLMInvoke", "harness"):
        assert name not in source


#: The exact set of native product modules allowed to import the public
#: ``llloom.okf`` surface. M002 allowed none. The M003 owner decision
#: authorizes validate-before-publish on the explicit opt-in creation path,
#: so exactly one module joins the set. This is an exact equality, not an
#: allow-list prefix, package glob, or "at least" condition.
OKF_NATIVE_IMPORTERS = {"src/llloom/ops/page.py"}


def test_okf_module_is_referenced_by_no_native_product_code():
    """Call-graph evidence: exactly the authorized native importer set imports it."""
    importers = []
    for path in sorted((REPO_ROOT / "src" / "llloom").rglob("*.py")):
        if path.parts[-2] == "okf":
            continue
        text = path.read_text(encoding="utf-8")
        if "llloom.okf" in text or "from .okf" in text:
            importers.append(path.relative_to(REPO_ROOT).as_posix())
    assert set(importers) == OKF_NATIVE_IMPORTERS
    assert len(importers) == len(OKF_NATIVE_IMPORTERS)


def test_the_authorized_native_importer_uses_a_direct_reviewable_import():
    """No dynamic construction, aliasing, or evasion of the isolation check."""
    source = (REPO_ROOT / "src" / "llloom" / "ops" / "page.py").read_text(
        encoding="utf-8"
    )
    assert "from llloom.okf import observe_page_okf" in source
    for evasion in ("import_module", "__import__", "importlib", "getattr(llloom"):
        assert evasion not in source

    tree = ast.parse(source)
    okf_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("llloom.okf")
    ]
    assert len(okf_imports) == 1
    assert [alias.name for alias in okf_imports[0].names] == ["observe_page_okf"]
    assert okf_imports[0].names[0].asname is None
    assert okf_imports[0].col_offset == 0, "the import is module level and reviewable"


def test_the_observer_runs_only_for_the_explicit_profiled_request(fresh_workspace):
    """Runtime proof: omitted requests never invoke the observer."""
    from llloom.ops import page as page_ops

    calls: list[str] = []
    real_observe = page_ops.observe_page_okf

    def counting(path):
        calls.append(str(path))
        return real_observe(path)

    original = page_ops.observe_page_okf
    page_ops.observe_page_okf = counting
    try:
        page_ops.create_page(fresh_workspace, page_id="concept/legacy-a", title="A")
        assert calls == [], "the legacy path invoked the observer"

        page_ops.create_page(
            fresh_workspace,
            page_id="concept/profiled-a",
            title="B",
            framework_profile="0.1-rc.1",
        )
        assert len(calls) == 1, "the profiled path must observe exactly once"

        page_ops.create_page(fresh_workspace, page_id="concept/legacy-b", title="C")
        assert len(calls) == 1, "a later omitted request invoked the observer"
    finally:
        page_ops.observe_page_okf = original


# ---- P01-P07 against exact, externally derived baselines ---------------------
#
# The CLI byte literals below were extracted once from the pinned llloom source
# commit 43ae4764d2d626d0626381b648c0d23658b11934 -- the commit M000 records --
# by `git archive`ing that commit into an external tree, installing it into an
# external venv, and running the real dispatcher for the scenario reproduced here.
# The extraction is recorded in the canonical self-report; nothing external is
# required at test runtime.

M000_LLLOOM_SOURCE_COMMIT = "43ae4764d2d626d0626381b648c0d23658b11934"
M000_EVIDENCE = (
    REPO_ROOT / "03_experiments" / "m000_llloom_native_baseline_evidence.md"
)

BASELINE_LIST_PAGES_JSON = """[
  {
    "page_id": "concept/demo",
    "page_path": "pages/concepts/demo.md",
    "page_class": "concept",
    "status": "draft",
    "write_policy": "mixed"
  },
  {
    "page_id": "overview",
    "page_path": "pages/overview.md",
    "page_class": "navigation",
    "status": "human_authored",
    "write_policy": "human"
  }
]
"""
BASELINE_LIST_PAGES_SHA = (
    "bd84e96e8bdef3e102756aace03ffecde7d4f5cd856e709678e0d54e23245669"
)
BASELINE_LIST_RENDER_TARGETS_JSON = "[]\n"
BASELINE_LIST_RENDER_TARGETS_SHA = (
    "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"
)

# The M000 evidence source content and the digest its query produced.
M000_EVIDENCE_SOURCE = (
    "# M000 Evidence\n\n"
    "The native llloom baseline preserves source-grounded evidence.\n"
)
M000_QUERY_EXCERPT_DIGEST = (
    "sha256:9f6029df47c9ca19df2bb7fcee3a82ec1791e7f740905433309d4ef35978a62a"
)

EXPECTED_RESULT_DATACLASSES = (
    "AcceptedDoctorWarning", "ClaimCard", "ClaimSummary", "CreatedClaim",
    "DoctorResult", "DoctorWarning", "EvidenceSummary", "HealthReport",
    "IngestResult", "LintResult", "MergeProposalSummary", "PageCreateResult",
    "PageSummary", "PdfPrepArtifact", "PdfPrepResult", "PlannedSeedClaim",
    "PromoteResult", "QueryResult", "ReconcileResult", "RenderPlanContributor",
    "RenderPlanEntry", "RenderResult", "RenderTargetListEntry",
    "RenderTargetSummary", "RetractResult", "SeedManifestResult",
    "SourceSummary", "StatusResult", "StructureItemHit", "SupersedeResult",
    "UnlockRecord", "UpdateReviewBundle", "VerbatimSpan", "VerifyResult",
)


def _run_cli(argv: list[str]) -> str:
    """Execute the real CLI dispatcher and capture its exact stdout bytes."""
    import contextlib
    import io

    from llloom.cli import main

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(argv)
    assert code == 0, argv
    return buffer.getvalue()


def _baseline_workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    _run_cli(["--root", str(root), "init"])
    _run_cli(["--root", str(root), "page", "create", "concept/demo",
              "--title", "Demo"])
    return root


def test_p01_p04_real_cli_json_bytes_match_the_pinned_source_baseline(tmp_path):
    """Executes `list-pages` and `list-render-targets` through the real dispatcher
    and compares exact bytes and key order with the externally derived baseline.
    A `json.dumps()` over library dataclasses is not CLI evidence."""
    root = _baseline_workspace(tmp_path)

    pages = _run_cli(["--root", str(root), "list-pages"])
    assert pages == BASELINE_LIST_PAGES_JSON
    assert hashlib.sha256(pages.encode("utf-8")).hexdigest() == \
        BASELINE_LIST_PAGES_SHA

    targets = _run_cli(["--root", str(root), "list-render-targets"])
    assert targets == BASELINE_LIST_RENDER_TARGETS_JSON
    assert hashlib.sha256(targets.encode("utf-8")).hexdigest() == \
        BASELINE_LIST_RENDER_TARGETS_SHA


def test_p01_p04_cli_bytes_are_unchanged_by_observing_every_page(tmp_path):
    root = _baseline_workspace(tmp_path)
    before = _run_cli(["--root", str(root), "list-pages"])
    (root / "pages" / "concepts" / "profiled.md").write_bytes(
        page(PROFILED).encode("utf-8")
    )
    for target in sorted((root / "pages").rglob("*.md")):
        observe_page_okf(target)
    # The new page is listed; the two baseline entries are unchanged, values and
    # key order alike.
    after = _run_cli(["--root", str(root), "list-pages"])
    baseline_entries = json.loads(BASELINE_LIST_PAGES_JSON)
    after_entries = json.loads(after)
    assert json.loads(before) == baseline_entries
    for entry in baseline_entries:
        match = next(a for a in after_entries if a["page_id"] == entry["page_id"])
        assert match == entry
        assert list(match.keys()) == list(entry.keys())
    assert _run_cli(["--root", str(root), "list-render-targets"]) == \
        BASELINE_LIST_RENDER_TARGETS_JSON


def test_p05_live_query_excerpt_digest_equals_the_accepted_m000_digest(
    fresh_workspace
):
    """The exact accepted digest, reproduced live from the M000 evidence content.
    A prose substring or a `sha256:` prefix is not an oracle."""
    ws = fresh_workspace
    source = ws.raw_sources / "m000-evidence.md"
    source.write_text(M000_EVIDENCE_SOURCE, encoding="utf-8")
    ingest(ws, source, source_class="raw_evidence")

    digests = {span.excerpt_hash
               for span in query(ws, question="evidence").used_verbatim_spans}
    assert digests == {M000_QUERY_EXCERPT_DIGEST}

    (ws.pages / "concepts" / "profiled.md").write_bytes(page(PROFILED).encode("utf-8"))
    for target in sorted(ws.pages.rglob("*.md")):
        observe_page_okf(target)
    after = {span.excerpt_hash
             for span in query(ws, question="evidence").used_verbatim_spans}
    assert after == {M000_QUERY_EXCERPT_DIGEST}


def test_p06_result_dataclass_inventory_is_exact_by_name():
    """The exact 34 names, not a prose search for the substring 34."""
    import llloom.ops.results as results

    declared = tuple(sorted(
        name for name in dir(results)
        if isinstance(getattr(results, name), type)
        and dataclasses.is_dataclass(getattr(results, name))
        and getattr(results, name).__module__ == "llloom.ops.results"
    ))
    assert declared == EXPECTED_RESULT_DATACLASSES
    assert len(declared) == 34


@pytest.mark.skipif(
    not M000_EVIDENCE.is_file(),
    reason="development-only M000 evidence is excluded from the release projection",
)
def test_p06_every_inventory_name_is_recorded_in_the_committed_m000_evidence():
    """The committed M000 evidence is the baseline oracle for the inventories."""
    evidence = M000_EVIDENCE.read_text(encoding="utf-8")
    for name in EXPECTED_RESULT_DATACLASSES:
        assert name in evidence, name
    for verb in sorted(EXPECTED_VERBS):
        assert verb in evidence, verb
    for tool in sorted(EXPECTED_MCP_TOOLS):
        assert tool in evidence, tool
    assert M000_QUERY_EXCERPT_DIGEST.split(":", 1)[1] in evidence
    assert M000_LLLOOM_SOURCE_COMMIT in evidence
