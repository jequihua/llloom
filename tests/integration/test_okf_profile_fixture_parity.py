"""Fixture parity for the OKF observation boundary.

Two independent corpora are exercised:

- the 24 pinned upstream fixtures, re-hashed from committed Git blob bytes
  against the release-contained digest manifest
  ``tests/fixtures/okf_profile/release_fixture_digests.json`` and compared with the
  fixture manifest's own ``full_parser`` expectations; and
- the llloom-owned L/H/P manifest, whose committed expectations were authored
  from the accepted M001 contract.

Neither oracle is computed by calling the implementation under test, and drift in
either corpus fails loudly rather than being absorbed.

The digest map ships with the corpus it protects, so this module collects and runs
wherever ``tests/fixtures`` is present. The frozen template pin manifest remains
the development authority: one dedicated cross-check compares the derived manifest
to that pin exactly, and only that check skips when the pin is absent, as it is by
design in the front-release projection.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from llloom.okf import observe_page_okf
from llloom.okf import observation as okf_observation

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "tests" / "fixtures" / "okf_profile"
LLLOOM_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "okf_pages"
#: Development-only authority. Excluded from the front-release projection by the
#: M006/S01 projection contract, so nothing at import or collection time may
#: require it.
PIN_MANIFEST = (
    REPO_ROOT / "04_delivery" / "llloom_okf_kickoff_handoff"
    / "template_candidate_pin_manifest.json"
)

#: Release-contained derived digest authority: the same 24 digests, carried by the
#: fixture corpus itself so the projected test plane is self-contained.
RELEASE_DIGESTS_PATH = UPSTREAM / "release_fixture_digests.json"
RELEASE_DIGESTS_SCHEMA = "llloom.release_fixture_digests.v1"

MARKERS = (
    "\n# T\n\n"
    "<!-- llloom:claim-block id=cb.t -->\n\n<!-- /llloom:claim-block -->\n\n"
    "<!-- llloom:commentary id=cm.t owner=human -->\n\n<!-- /llloom:commentary -->\n"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


UPSTREAM_MANIFEST = _load(UPSTREAM / "manifest.json")
LLLOOM_MANIFEST = _load(LLLOOM_FIXTURES / "manifest.json")
RELEASE_DIGESTS = _load(RELEASE_DIGESTS_PATH)

#: The digest map every gate below uses. It is release-contained on purpose; the
#: frozen pin is compared against it by one development-authority test.
PINNED_DIGESTS = {
    entry["path"]: entry["sha256"] for entry in RELEASE_DIGESTS["fixtures"]
}

PIN_MANIFEST_PRESENT = PIN_MANIFEST.is_file()
requires_frozen_pin = pytest.mark.skipif(
    not PIN_MANIFEST_PRESENT,
    reason="development-only frozen template pin manifest is excluded from the "
           "release projection",
)


def committed_blob(repo_relative: str) -> bytes | None:
    """Bytes of the committed blob at HEAD, or None outside a Git checkout."""
    result = subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{repo_relative}"],
        cwd=REPO_ROOT, capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


# ---- the release-contained digest authority -----------------------------------


def test_the_release_digest_manifest_declares_its_schema_and_provenance():
    """The derived manifest names what it is and where it came from."""
    assert RELEASE_DIGESTS["schema"] == RELEASE_DIGESTS_SCHEMA
    assert RELEASE_DIGESTS["digest_basis"] == "committed_git_blob_bytes"
    assert RELEASE_DIGESTS["template_baseline_commit"] ==         "97ab4cd6b4f6f890d81c2738885051f2c4f79bc4"
    provenance = RELEASE_DIGESTS["source_pin_manifest_sha256"]
    assert len(provenance) == 64 and provenance == provenance.lower()
    assert set(RELEASE_DIGESTS) == {
        "schema", "template_baseline_commit", "source_pin_manifest_sha256",
        "digest_basis", "fixtures",
    }, "the derived manifest carries only digest provenance, never outcomes"


def test_the_release_digest_manifest_covers_exactly_the_24_fixture_paths():
    paths = [entry["path"] for entry in RELEASE_DIGESTS["fixtures"]]
    assert len(paths) == 24
    assert paths == sorted(paths), "records must be path-sorted"
    assert len(set(paths)) == 24
    assert paths == sorted(f["path"] for f in UPSTREAM_MANIFEST["fixtures"])
    for path in paths:
        assert path.startswith("tests/fixtures/okf_profile/")
        assert path.endswith(".md")


def test_every_release_digest_is_a_lowercase_sha256():
    for entry in RELEASE_DIGESTS["fixtures"]:
        digest = entry["sha256"]
        assert len(digest) == 64, entry["path"]
        assert digest == digest.lower(), entry["path"]
        assert set(digest) <= set("0123456789abcdef"), entry["path"]
        assert set(entry) == {"path", "sha256"}, entry["path"]


def test_every_release_digest_matches_the_current_fixture_bytes():
    """The release-contained map is true of the corpus that ships with it."""
    for entry in RELEASE_DIGESTS["fixtures"]:
        data = (REPO_ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]


@requires_frozen_pin
def test_the_release_digest_manifest_equals_the_frozen_development_pin():
    """Development authority: the derived map is exactly the frozen pin's fixtures.

    This is the one check that needs the excluded governance input, so it is the
    one check that skips in the release projection.
    """
    pin_bytes = PIN_MANIFEST.read_bytes()
    assert hashlib.sha256(pin_bytes).hexdigest() ==         RELEASE_DIGESTS["source_pin_manifest_sha256"], (
            "the derived manifest was not generated from this frozen pin"
        )
    pin = json.loads(pin_bytes.decode("utf-8"))
    frozen = sorted(
        ({"path": record["path"], "sha256": record["sha256"].lower()}
         for record in pin["files"] if record.get("role") == "fixture"),
        key=lambda record: record["path"],
    )
    assert frozen == RELEASE_DIGESTS["fixtures"], (
        "the release-contained digests drifted from the frozen template pin"
    )
    assert pin["template_baseline"]["commit"] ==         RELEASE_DIGESTS["template_baseline_commit"]


# ---- the 24 pinned upstream fixtures ------------------------------------------


def test_upstream_corpus_declares_the_pinned_fixture_count():
    assert len(UPSTREAM_MANIFEST["fixtures"]) == 24
    assert UPSTREAM_MANIFEST["profile_candidate"] == "0.1-rc.1"


def verified_snapshot(repo_relative: str, root: Path = REPO_ROOT):
    """Verify the checkout bytes, then freeze them into the parse core's own input.

    The gate reads the file **once**, hashes exactly those bytes, and hands the same
    bytes to `llloom.okf.observation._snapshot_from_bytes`. The snapshot is what the
    parity observation parses, through `_observe_snapshot` -- the identical core the
    public `observe_page_okf(path)` wrapper calls after its own bounded read.

    That removes the interval the correction review exploited: an earlier gate
    hashed the bytes and returned a mutable `Path`, so a replacement between the
    digest check and the product's own `open()` was consumed unnoticed. Here there
    is no second read to substitute into.

    The read is also bounded. One binary open and one `read(ceiling + 1)`: a
    checkout that has drifted to an enormous size is rejected on size **before**
    anything hashes or parses it, so a Level 4 provenance gate cannot be turned into
    unbounded checkout-controlled memory and I/O by a file it is meant to police.
    """
    ceiling = okf_observation.MAX_ARTIFACT_BYTES
    request = ceiling + 1
    with open(root / repo_relative, "rb") as handle:
        data = handle.read(request)
    if len(data) > ceiling:
        # Before SHA-256, before `_snapshot_from_bytes`, before the pin lookup.
        raise AssertionError(
            f"checkout bytes of {repo_relative} exceed the {ceiling}-byte ceiling; "
            f"at most {request} bytes were requested and retained"
        )
    digest = hashlib.sha256(data).hexdigest()
    assert digest == PINNED_DIGESTS[repo_relative], (
        f"checkout bytes of {repo_relative} drifted from the frozen pin: "
        f"{digest} != {PINNED_DIGESTS[repo_relative]}"
    )
    return okf_observation._snapshot_from_bytes(data)


def observe_verified(repo_relative: str, root: Path = REPO_ROOT):
    """The one parity observation path: verified bytes into the shared parse core."""
    return okf_observation._observe_snapshot(verified_snapshot(repo_relative, root))


@pytest.mark.parametrize(
    "fixture", UPSTREAM_MANIFEST["fixtures"],
    ids=[f["id"] for f in UPSTREAM_MANIFEST["fixtures"]],
)
def test_upstream_fixture_digest_has_not_drifted(fixture):
    """Provenance gate: a changed pinned fixture is a stop condition, never an
    expectation to update locally. Both the committed blob and the checkout bytes
    must match the pin, because only the latter is what the product reads."""
    blob = committed_blob(fixture["path"])
    if blob is not None:
        assert hashlib.sha256(blob).hexdigest() == PINNED_DIGESTS[fixture["path"]], (
            f"committed blob {fixture['path']} drifted from its recorded digest"
        )
    verified_snapshot(fixture["path"])


@pytest.mark.parametrize(
    "fixture", UPSTREAM_MANIFEST["fixtures"],
    ids=[f["id"] for f in UPSTREAM_MANIFEST["fixtures"]],
)
def test_upstream_fixture_reproduces_its_pinned_full_parser_outcome(fixture):
    expected = fixture["expected"]["full_parser"]
    observed = observe_verified(fixture["path"])
    assert observed.okf_concept_result == expected["okf_concept"]["result"]
    assert observed.okf_concept_reason == expected["okf_concept"]["reason"]
    assert observed.framework_profile_result == expected["framework_profile"]["result"]
    assert observed.framework_profile_reason == expected["framework_profile"]["reason"]
    assert observed.execution_eligibility == fixture["expected"]["execution_eligibility"]
    assert observed.execution_eligibility == "not_evaluated"


def test_the_public_path_wrapper_and_the_gate_share_one_parse_core():
    """The gate is not a second implementation: both routes end in the same core."""
    import inspect

    wrapper = inspect.getsource(okf_observation.observe_page_okf)
    assert "_observe_snapshot" in wrapper
    assert "_read_snapshot" in wrapper
    for fixture in UPSTREAM_MANIFEST["fixtures"][:5]:
        path = REPO_ROOT / fixture["path"]
        assert observe_verified(fixture["path"]) == observe_page_okf(path)


def test_okf_yaml_unsupported_is_never_emitted_on_the_full_parser_lane():
    """The subset lane uses that code for seven fixtures; llloom consumes the full
    lane, never emits it, and must tolerate it if an upstream producer does."""
    subset_users = [
        f["id"] for f in UPSTREAM_MANIFEST["fixtures"]
        if f["expected"]["subset_parser"]["okf_concept"]["reason"]
        == "OKF_YAML_UNSUPPORTED"
    ]
    assert len(subset_users) == 7
    for fixture in UPSTREAM_MANIFEST["fixtures"]:
        assert observe_verified(fixture["path"]).okf_concept_reason \
            != "OKF_YAML_UNSUPPORTED"


FIXTURE_UNDER_MUTATION = "tests/fixtures/okf_profile/legacy_no_frontmatter.md"


def _test_owned_checkout(tmp_path: Path) -> Path:
    """A complete, test-owned checkout root. The authoritative corpus is never
    mutated, restored, or even opened for writing."""
    root = tmp_path / "checkout"
    (root / "tests" / "fixtures" / "okf_profile").mkdir(parents=True)
    for fixture in UPSTREAM_MANIFEST["fixtures"]:
        destination = root / fixture["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / fixture["path"]).read_bytes())
    return root


SEMANTIC_PRESERVING_SUFFIX = b"\n<!-- semantic-preserving comment -->\n"


def test_drift_before_the_gate_fails_the_real_gate(tmp_path):
    """Interval one: a checkout edit made before verification must fail."""
    root = _test_owned_checkout(tmp_path)
    target = root / FIXTURE_UNDER_MUTATION
    original = target.read_bytes()
    assert verified_snapshot(FIXTURE_UNDER_MUTATION, root)      # clean copy passes

    target.write_bytes(original + SEMANTIC_PRESERVING_SUFFIX)
    with pytest.raises(AssertionError, match="drifted from the frozen pin"):
        verified_snapshot(FIXTURE_UNDER_MUTATION, root)

    # The edit is byte-level only: the outcome would have been unchanged, which is
    # exactly why an outcome-only comparison absorbed it before.
    mutated = okf_observation._snapshot_from_bytes(target.read_bytes())
    clean = okf_observation._snapshot_from_bytes(original)
    assert okf_observation._observe_snapshot(mutated) == \
        okf_observation._observe_snapshot(clean)
    assert (REPO_ROOT / FIXTURE_UNDER_MUTATION).read_bytes() == original


def test_replacement_after_the_gate_cannot_reach_the_observation(tmp_path):
    """Interval two: once verified, the bytes are frozen. A post-gate replacement
    changes nothing about the computed outcome, and a fresh gate call fails."""
    root = _test_owned_checkout(tmp_path)
    target = root / FIXTURE_UNDER_MUTATION
    original = target.read_bytes()

    snapshot = verified_snapshot(FIXTURE_UNDER_MUTATION, root)
    from_verified = okf_observation._observe_snapshot(snapshot)

    # Replace the checkout file after the gate passed but before the comparison.
    target.write_bytes(original + SEMANTIC_PRESERVING_SUFFIX)
    assert hashlib.sha256(target.read_bytes()).hexdigest() != \
        PINNED_DIGESTS[FIXTURE_UNDER_MUTATION]

    # The observation still comes from the verified bytes, not the changed path.
    assert okf_observation._observe_snapshot(snapshot) == from_verified
    assert snapshot.text == original.decode("utf-8")

    # And a new gate invocation against the changed checkout fails.
    with pytest.raises(AssertionError, match="drifted from the frozen pin"):
        verified_snapshot(FIXTURE_UNDER_MUTATION, root)

    assert (REPO_ROOT / FIXTURE_UNDER_MUTATION).read_bytes() == original


# ---- the gate's own read is bounded ------------------------------------------

# The accepted total-input ceiling and the bounded read size, as independent
# literals. The gate must use the product constant; this oracle must not.
ACCEPTED_ARTIFACT_CEILING = 1_048_576
ACCEPTED_READ_REQUEST = 1_048_577


def test_the_product_ceiling_equals_the_accepted_literal():
    assert okf_observation.MAX_ARTIFACT_BYTES == ACCEPTED_ARTIFACT_CEILING


def test_the_real_gate_performs_one_bounded_binary_read(tmp_path, monkeypatch):
    """Instrument the real gate's open/read boundary: one binary open, one read,
    and the exact request 1,048,577."""
    root = _test_owned_checkout(tmp_path)
    opens: list[tuple] = []
    reads: list[int] = []
    real_open = builtins.open

    class _Recording:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

        def read(self, size=-1):
            reads.append(size)
            return self._handle.read(size)

    def recording_open(path, mode="r", *args, **kwargs):
        if str(path).endswith(".md") and "b" in mode:
            opens.append((str(path), mode))
            return _Recording(real_open(path, mode, *args, **kwargs))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    snapshot = verified_snapshot(FIXTURE_UNDER_MUTATION, root)

    assert len(opens) == 1, opens
    assert opens[0][1] == "rb"
    assert reads == [ACCEPTED_READ_REQUEST]
    assert snapshot.status == "ok"


def test_the_real_gate_rejects_an_over_ceiling_checkout(tmp_path):
    """A 4 MiB checkout payload is rejected, and at most ceiling-plus-one bytes are
    ever requested or retained."""
    root = tmp_path / "oversized"
    target = root / FIXTURE_UNDER_MUTATION
    target.parent.mkdir(parents=True)
    payload = b"a" * (4 * 1024 * 1024)
    target.write_bytes(payload)
    assert target.stat().st_size == 4 * 1024 * 1024

    with pytest.raises(AssertionError, match="exceed the 1048576-byte ceiling"):
        verified_snapshot(FIXTURE_UNDER_MUTATION, root)


def test_over_ceiling_rejection_happens_before_hashing_or_parsing(tmp_path,
                                                                 monkeypatch):
    """Rejection must precede SHA-256, snapshot construction, and the parse core --
    not merely produce an eventual digest mismatch."""
    root = tmp_path / "oversized"
    target = root / FIXTURE_UNDER_MUTATION
    target.parent.mkdir(parents=True)
    target.write_bytes(b"b" * (4 * 1024 * 1024))

    def forbidden(*args, **kwargs):
        raise AssertionError("the gate hashed or parsed an over-ceiling checkout")

    monkeypatch.setattr(hashlib, "sha256", forbidden)
    monkeypatch.setattr(okf_observation, "_snapshot_from_bytes", forbidden)
    monkeypatch.setattr(okf_observation, "_observe_snapshot", forbidden)

    with pytest.raises(AssertionError, match="exceed the 1048576-byte ceiling"):
        verified_snapshot(FIXTURE_UNDER_MUTATION, root)


def test_a_payload_at_exactly_the_ceiling_is_still_read_and_gated(tmp_path):
    """The bound is on size, not on content: a file exactly at the ceiling is read
    in full and then fails the pin, proving the rejection is not off by one."""
    root = tmp_path / "atmax"
    target = root / FIXTURE_UNDER_MUTATION
    target.parent.mkdir(parents=True)
    target.write_bytes(b"c" * ACCEPTED_ARTIFACT_CEILING)
    assert target.stat().st_size == ACCEPTED_ARTIFACT_CEILING

    with pytest.raises(AssertionError, match="drifted from the frozen pin"):
        verified_snapshot(FIXTURE_UNDER_MUTATION, root)


def test_the_authoritative_corpus_is_never_written_by_a_gate_test():
    """Every mutation in this module happens in a disposable test-owned root."""
    for fixture in UPSTREAM_MANIFEST["fixtures"]:
        data = (REPO_ROOT / fixture["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == PINNED_DIGESTS[fixture["path"]]


# ---- the llloom-owned corpus ---------------------------------------------------


COMMITTED_CASES = [c for c in LLLOOM_MANIFEST["cases"] if c["kind"] == "committed_bytes"]
# L19 (R0U) needs an injected OSError rather than a file on disk, so it has its
# own test below instead of a generic recipe.
CONSTRUCTED_CASES = [
    c for c in LLLOOM_MANIFEST["cases"]
    if c["kind"] == "constructed" and c["row"] != "R0U"
]
R0U_CASES = [c for c in LLLOOM_MANIFEST["cases"] if c["row"] == "R0U"]


@pytest.mark.parametrize(
    "case", COMMITTED_CASES, ids=[c["id"] for c in COMMITTED_CASES],
)
def test_llloom_fixture_bytes_match_their_recorded_digest(case):
    data = (REPO_ROOT / case["path"]).read_bytes()
    assert len(data) == case["bytes"]
    assert hashlib.sha256(data).hexdigest() == case["sha256"], (
        f"llloom fixture {case['id']} drifted from its recorded digest"
    )


@pytest.mark.parametrize(
    "case", COMMITTED_CASES, ids=[c["id"] for c in COMMITTED_CASES],
)
def test_llloom_committed_case_reproduces_its_expected_observation(case):
    expected = case["expected_observation"]
    observed = observe_page_okf(REPO_ROOT / case["path"])
    for field, want in expected.items():
        assert getattr(observed, field) == want, f"{case['id']}.{field}"


def _build_constructed(case_id: str, tmp_path: Path) -> Path:
    """Deterministic recipes recorded in the llloom manifest."""
    from llloom.ops.page import _render_stub
    from llloom.workspace.layout import _STARTER_OVERVIEW

    pages = tmp_path / "pages" / "concepts"
    pages.mkdir(parents=True, exist_ok=True)
    if case_id == "L02":
        text = _render_stub(
            normalized_page_id="concept/m000-demo", page_class="concept",
            title="M000 Demo", claim_block_id="claim_block.concept.m000-demo",
            commentary_id="commentary.concept.m000-demo",
        )
        target = pages / "m000-demo.md"
        target.write_bytes(text.encode("utf-8"))
        return target
    if case_id == "L03":
        target = pages / "overview.md"
        target.write_bytes(_STARTER_OVERVIEW.encode("utf-8"))
        return target
    if case_id == "L13":
        source = (REPO_ROOT / "tests/fixtures/okf_pages/L04_profiled_page.md")
        target = pages / "crlf.md"
        target.write_bytes(source.read_text(encoding="utf-8").replace("\n", "\r\n")
                           .encode("utf-8"))
        return target
    if case_id == "L18":
        return pages / "never_created.md"
    raise AssertionError(f"no recipe for {case_id}")


@pytest.mark.parametrize(
    "case", CONSTRUCTED_CASES, ids=[c["id"] for c in CONSTRUCTED_CASES],
)
def test_llloom_constructed_case_reproduces_its_expected_observation(case, tmp_path):
    target = _build_constructed(case["id"], tmp_path)
    observed = observe_page_okf(target)
    for field, want in case["expected_observation"].items():
        assert getattr(observed, field) == want, f"{case['id']}.{field}"


def test_l02_and_l03_reproduce_their_recorded_m000_byte_goldens(tmp_path):
    """The M000 evidence recorded the Windows CRLF byte form of both pages; the
    same text under LF is the platform-independent counterpart."""
    for case in CONSTRUCTED_CASES:
        if "m000_crlf_sha256" not in case:
            continue
        target = _build_constructed(case["id"], tmp_path)
        text = target.read_text(encoding="utf-8")
        crlf = hashlib.sha256(text.replace("\n", "\r\n").encode("utf-8")).hexdigest()
        assert crlf == case["m000_crlf_sha256"], case["id"]


def test_llloom_manifest_marker_expectations_match_the_native_parser(tmp_path):
    """Column 5 is an observation, so the manifest's marker values are checked
    against llloom's own region parser rather than assumed."""
    from llloom.pages.regions import PageParseError, parse_page

    def observed_marker(path: Path, in_scope: bool) -> str:
        if not path.exists() or not path.is_file():
            return "missing_page"
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            return "not_observed"
        if not in_scope:
            return "not_observed"
        try:
            parse_page(text)
        except PageParseError:
            return "parse_error"
        except Exception:  # noqa: BLE001
            return "native_error"
        return "ok"

    for case in COMMITTED_CASES + CONSTRUCTED_CASES:
        expected = case["expected_columns"]["marker_health"]
        if case["kind"] == "committed_bytes":
            target = REPO_ROOT / case["path"]
        else:
            target = _build_constructed(case["id"], tmp_path)
        in_scope = case["marker_class"] in ("M1", "M2")
        assert observed_marker(target, in_scope) == expected, case["id"]


def test_llloom_hostile_manifest_declares_both_sides_of_every_numeric_ceiling():
    hostile = {case["id"]: case for case in LLLOOM_MANIFEST["hostile_cases"]}
    for prefix in ("H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H09",
                   "H10", "H11"):
        assert f"{prefix}-max" in hostile
        assert f"{prefix}-plus1" in hostile
        assert hostile[f"{prefix}-plus1"]["expected_observation"][
            "okf_concept_reason"] == "OKF_PARSE_LIMIT_EXCEEDED"
        assert hostile[f"{prefix}-max"]["expected_observation"][
            "okf_concept_reason"] != "OKF_PARSE_LIMIT_EXCEEDED"
    assert hostile["H05-max"]["row"] == "R7"
    assert hostile["H05-plus1"]["row"] == "R6"
    assert "9983" in hostile["H05-max"]["construction"]
    assert "9984" in hostile["H05-plus1"]["construction"]
    for supplemental in ("H05-nearmax", "H05-order"):
        assert "supplemental" in hostile[supplemental]["ceiling"]
    assert hostile["H12-cycle"]["row"] == "R6"
    assert hostile["H13-recursion"]["expected_columns"]["marker_health"] == \
        "native_error"


def test_llloom_hostile_cases_execute_to_their_declared_rows(tmp_path):
    """Every hostile case is rebuilt from its manifest recipe and observed."""
    from tests.unit.test_okf_page_observation import (  # noqa: PLC0415
        ROW_SHAPE, aliases, chunked_flow, comma_block, flow_mapping, folded_scalar,
        frontmatter_of_bytes, nested_mapping, page, shape,
    )
    from llloom.okf import observation as obs

    base = page('type: analysis\nframework_profile: "0.1-rc.1"')
    pad = obs.MAX_ARTIFACT_BYTES - len(base.encode("utf-8"))
    builders = {
        "H01-max": base + "a" * pad,
        "H01-plus1": base + "a" * (pad + 1),
        "H02-max": page(frontmatter_of_bytes(obs.MAX_FRONTMATTER_BYTES)),
        "H02-plus1": page(frontmatter_of_bytes(obs.MAX_FRONTMATTER_BYTES + 1)),
        "H03-max": page('type: analysis\nframework_profile: "0.1-rc.1"\n'
                        + "\n".join(f'k{i}: "v"' for i in range(498))),
        "H03-plus1": page('type: analysis\nframework_profile: "0.1-rc.1"\n'
                          + "\n".join(f'k{i}: "v"' for i in range(499))),
        "H04-max": page('type: analysis\nframework_profile: "0.1-rc.1"\nnote: "'
                        + "a" * (obs.MAX_LINE_LEN - 8) + '"'),
        "H04-plus1": page('type: analysis\nframework_profile: "0.1-rc.1"\nnote: "'
                          + "a" * (obs.MAX_LINE_LEN - 7) + '"'),
        "H05-max": page(comma_block(9983)),
        "H05-plus1": page(comma_block(9984)),
        "H05-nearmax": page(chunked_flow(1988)),
        "H05-order": page(chunked_flow(4993) + "\nbad:\n\ttabbed: 1"),
        "H06-max": page(chunked_flow(1989)),
        "H06-plus1": page(chunked_flow(1990)),
        "H07-max": page(nested_mapping(32)),
        "H07-plus1": page(nested_mapping(33)),
        "H08-max": page(folded_scalar(obs.MAX_SCALAR_LEN)),
        "H08-plus1": page(folded_scalar(obs.MAX_SCALAR_LEN + 1)),
        "H09-max": page(flow_mapping(obs.MAX_MAPPING_ITEMS)),
        "H09-plus1": page(flow_mapping(obs.MAX_MAPPING_ITEMS + 1)),
        "H10-max": page(chunked_flow(obs.MAX_SEQUENCE_ITEMS, chunk=obs.MAX_SEQUENCE_ITEMS)),
        "H10-plus1": page(chunked_flow(obs.MAX_SEQUENCE_ITEMS + 1,
                                       chunk=obs.MAX_SEQUENCE_ITEMS + 1)),
        "H11-max": page(aliases(obs.MAX_ALIASES)),
        "H11-plus1": page(aliases(obs.MAX_ALIASES + 1)),
        "H12-cycle": page('type: analysis\nframework_profile: "0.1-rc.1"'
                          "\ncycle: &c\n  self: *c"),
        "H13-recursion": page('type: analysis\nframework_profile: "0.1-rc.1"\nx: '
                              + "[" * 1500 + "]" * 1500),
    }
    hostile = {case["id"]: case for case in LLLOOM_MANIFEST["hostile_cases"]}
    assert set(builders) == set(hostile)

    for case_id, text in builders.items():
        target = tmp_path / f"{case_id}.md"
        target.write_bytes(text.encode("utf-8"))
        observed = observe_page_okf(target)
        assert shape(observed) == ROW_SHAPE[hostile[case_id]["row"]], case_id
        for field, want in hostile[case_id]["expected_observation"].items():
            assert getattr(observed, field) == want, f"{case_id}.{field}"


@pytest.mark.parametrize("case", R0U_CASES, ids=[c["id"] for c in R0U_CASES])
def test_llloom_r0u_case_reproduces_its_expected_observation(case, tmp_path,
                                                             monkeypatch):
    """R0U: a present regular target whose bytes cannot be read."""
    target = tmp_path / "present_but_denied.md"
    newline = chr(10)
    target.write_bytes(
        ("---" + newline + "type: analysis" + newline + "---" + newline)
        .encode("utf-8")
    )
    real_stat = Path.stat

    def denied(self, *args, **kwargs):
        if self.name == "present_but_denied.md":
            raise PermissionError(13, "permission denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    observed = observe_page_okf(target)
    for field, want in case["expected_observation"].items():
        assert getattr(observed, field) == want, f"{case['id']}.{field}"
    assert case["marker_class"] == "M3"
    assert case["expected_columns"]["marker_health"] == "not_observed"
