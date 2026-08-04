"""Unit tests for the bounded read-only OKF observation.

Expectations are the accepted M001 contract values, written literally here. No
test computes its oracle by calling the implementation under test.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest
import yaml

from llloom.okf import PageOkfObservation, observe_page_okf
from llloom.okf import observation as obs

MARKERS = (
    "\n# T\n\n"
    "<!-- llloom:claim-block id=cb.t -->\n\n<!-- /llloom:claim-block -->\n\n"
    "<!-- llloom:commentary id=cm.t owner=human -->\n\n<!-- /llloom:commentary -->\n"
)
PROFILED = 'type: analysis\nframework_profile: "0.1-rc.1"'

# The eleven accepted safety ceilings, written as literals. This is the test oracle:
# it must never be read from `llloom.okf.observation`, or a one-byte weakening of a
# declared boundary would move the expectation along with it. The same values are
# recorded in tests/fixtures/okf_pages/manifest.json under "safety_constants".
ACCEPTED = {
    "max_artifact_bytes": 1_048_576,
    "max_frontmatter_bytes": 65_536,
    "max_frontmatter_lines": 500,
    "max_line_length": 8_192,
    "max_tokens": 10_000,
    "max_nodes": 2_000,
    "max_depth": 32,
    "max_scalar_length": 16_384,
    "max_mapping_items": 500,
    "max_sequence_items": 1_000,
    "max_aliases": 50,
}
IMPLEMENTATION_CONSTANT = {
    "max_artifact_bytes": "MAX_ARTIFACT_BYTES",
    "max_frontmatter_bytes": "MAX_FRONTMATTER_BYTES",
    "max_frontmatter_lines": "MAX_FRONTMATTER_LINES",
    "max_line_length": "MAX_LINE_LEN",
    "max_tokens": "MAX_TOKENS",
    "max_nodes": "MAX_NODES",
    "max_depth": "MAX_DEPTH",
    "max_scalar_length": "MAX_SCALAR_LEN",
    "max_mapping_items": "MAX_MAPPING_ITEMS",
    "max_sequence_items": "MAX_SEQUENCE_ITEMS",
    "max_aliases": "MAX_ALIASES",
}


@pytest.mark.parametrize("key,name", sorted(IMPLEMENTATION_CONSTANT.items()))
def test_implementation_constant_matches_the_accepted_value(key, name):
    """A ceiling that drifts from the accepted contract fails here first."""
    assert getattr(obs, name) == ACCEPTED[key], name


def test_manifest_records_the_same_accepted_constants():
    import json

    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures" / "okf_pages"
         / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["safety_constants"] == ACCEPTED


# The seven public fields, in declaration order, for every accepted row.
ROW_SHAPE: dict[str, tuple] = {
    "R0": ("target_absent", "not_evaluated", None, "not_applicable", None,
           "not_evaluated", None),
    "R1": ("oversize", "unverified", "OKF_PARSE_LIMIT_EXCEEDED", "fail",
           "PROFILE_YAML_OUT_OF_SUBSET", "not_evaluated", None),
    "R2": ("undecodable", "not_evaluated", None, "not_applicable", None,
           "not_evaluated", None),
    "R3": ("ok", "not_evaluated", None, "not_applicable", None, "not_evaluated", None),
    "R4": ("ok", "fail", "OKF_FRONTMATTER_MISSING", "not_applicable", None,
           "not_evaluated", None),
    "R5": ("ok", "unverified", "OKF_PARSE_LIMIT_EXCEEDED", "fail",
           "PROFILE_YAML_OUT_OF_SUBSET", "not_evaluated", None),
    "R6": ("ok", "unverified", "OKF_PARSE_LIMIT_EXCEEDED", "fail",
           "PROFILE_YAML_OUT_OF_SUBSET", "not_evaluated", None),
    "R7": ("ok", "fail", "OKF_YAML_INVALID", "fail", "PROFILE_YAML_OUT_OF_SUBSET",
           "not_evaluated", None),
    "R8": ("ok", "fail", "OKF_YAML_INVALID", "fail", "PROFILE_YAML_OUT_OF_SUBSET",
           "not_evaluated", None),
    "R9": ("ok", "fail", "OKF_TYPE_MISSING", "not_applicable", None,
           "not_evaluated", None),
    "R10": ("ok", "pass", None, "fail", "PROFILE_YAML_OUT_OF_SUBSET",
            "not_evaluated", None),
    "R11": ("ok", "pass", None, "fail", "PROFILE_TYPE_UNSUPPORTED",
            "not_evaluated", None),
    "R12": ("ok", "pass", None, "not_applicable", None, "not_evaluated", None),
    "R13": ("ok", "pass", None, "fail", "PROFILE_VERSION_UNSUPPORTED",
            "not_evaluated", "9.9-rc.9"),
    "R14": ("ok", "pass", None, "pass", None, "not_evaluated", "0.1-rc.1"),
}


def shape(observation: PageOkfObservation) -> tuple:
    return (
        observation.read_status,
        observation.okf_concept_result,
        observation.okf_concept_reason,
        observation.framework_profile_result,
        observation.framework_profile_reason,
        observation.execution_eligibility,
        observation.declared_framework_profile,
    )


def page(body: str, markers: str = MARKERS) -> str:
    return "---\n" + body + "\n---\n" + markers


def write(tmp_path: Path, name: str, data: str | bytes) -> Path:
    target = tmp_path / name
    target.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return target


# ---- deterministic constructions for the accepted boundary cases -------------


def chunked_flow(total_items: int, chunk: int = 900, item: str = '"a"') -> str:
    """Flow sequences of at most ``chunk`` items, so the per-sequence ceiling
    never wins before the ceiling under test."""
    parts, done, index = [], 0, 0
    while done < total_items:
        take = min(chunk, total_items - done)
        parts.append(_flow_sequence(f"p{index}", [item] * take))
        done += take
        index += 1
    return PROFILED + "\n" + "\n".join(parts)


def _flow_sequence(key: str, items: list[str], width: int = 7000) -> str:
    out, line = [], f"{key}: ["
    for n, value in enumerate(items):
        piece = value + ("," if n < len(items) - 1 else "")
        if len(line) + len(piece) + 1 > width:
            out.append(line)
            line = "  "
        line += piece
    out.append(line + "]")
    return "\n".join(out)


def flow_mapping(entries: int, width: int = 7000) -> str:
    out, line = [], "pad: {"
    for i in range(entries):
        piece = f'k{i}: "v"' + ("," if i < entries - 1 else "")
        if len(line) + len(piece) + 1 > width:
            out.append(line)
            line = "  "
        line += piece
    out.append(line + "}")
    return PROFILED + "\n" + "\n".join(out)


def nested_mapping(levels: int) -> str:
    lines = ["  " * i + f"n{i}:" for i in range(levels - 1)]
    lines.append("  " * (levels - 1) + 'leaf: "v"')
    return PROFILED + "\n" + "\n".join(lines)


def folded_scalar(total: int, chunks: int = 3) -> str:
    gaps = chunks - 1
    base = (total - gaps) // chunks
    sizes = [base] * chunks
    sizes[-1] += (total - gaps) - base * chunks
    body = 'note: "' + sizes[0] * "a"
    for size in sizes[1:]:
        body += "\n  " + size * "a"
    return PROFILED + "\n" + body + '"'


def frontmatter_of_bytes(target: int) -> str:
    """A frontmatter block of exactly ``target`` ASCII bytes."""
    lines, cur, index = [PROFILED], len(PROFILED), 0
    while True:
        candidate = 'k%03d: "%s"' % (index, "a" * 6000)
        if cur + 1 + len(candidate) + 1 + 32 > target:
            break
        lines.append(candidate)
        cur += 1 + len(candidate)
        index += 1
    head = 'tail: "'
    lines.append(head + "a" * (target - cur - 1 - len(head) - 1) + '"')
    return "\n".join(lines)


def comma_block(commas: int, width: int = 6000) -> str:
    """The accepted scanned-token construction: a closed but invalid flow
    sequence carrying only entry separators."""
    lines = ['type: page', 'framework_profile: "0.1-rc.1"', "data: ["]
    left = commas
    while left > 0:
        take = min(width, left)
        lines.append("," * take)
        left -= take
    lines.append("]")
    return "\n".join(lines)


def scanned_tokens(frontmatter: str) -> int:
    return sum(1 for _ in yaml.scan(frontmatter, Loader=yaml.SafeLoader))


def aliases(count: int) -> str:
    return (PROFILED + '\nanchor: &a "v"\n'
            + _flow_sequence("pad", ["*a"] * count))


# ---- every accepted row, with its exact seven-field shape --------------------

ROW_CASES = [
    ("R3", lambda t: write(t, "legacy.md", "# Legacy\n\ntext\n" + MARKERS)),
    ("R4", lambda t: write(t, "unterminated.md",
                           "---\ntype: analysis\n\n# u\n" + MARKERS)),
    ("R5", lambda t: write(t, "fm_bytes.md",
                           page(frontmatter_of_bytes(ACCEPTED["max_frontmatter_bytes"] + 1)))),
    ("R6", lambda t: write(t, "depth.md", page(nested_mapping(33)))),
    ("R7", lambda t: write(t, "invalid.md", page("type: analysis\n\tbad: 1"))),
    ("R8", lambda t: write(t, "dupe.md", page("type: analysis\ntype: other"))),
    ("R9", lambda t: write(t, "no_type.md", page('title: "x"'))),
    ("R10", lambda t: write(t, "flow.md", page('type: analysis\ntags: ["a"]'))),
    ("R11", lambda t: write(t, "unknown_type.md", page('type: "Made Up Type"'))),
    ("R12", lambda t: write(t, "no_profile.md", page('type: analysis\ntitle: "x"'))),
    ("R13", lambda t: write(t, "bad_version.md",
                            page('type: analysis\nframework_profile: "9.9-rc.9"'))),
    ("R14", lambda t: write(t, "valid.md", page(PROFILED))),
]


@pytest.mark.parametrize("row,build", ROW_CASES, ids=[r for r, _ in ROW_CASES])
def test_row_returns_the_accepted_seven_field_shape(tmp_path, row, build):
    assert shape(observe_page_okf(build(tmp_path))) == ROW_SHAPE[row]


def test_r0_absent_target(tmp_path):
    assert shape(observe_page_okf(tmp_path / "never_created.md")) == ROW_SHAPE["R0"]


def test_r0_non_regular_target(tmp_path):
    directory = tmp_path / "a_directory.md"
    directory.mkdir()
    assert shape(observe_page_okf(directory)) == ROW_SHAPE["R0"]


def test_r1_oversize_refused_before_decode(tmp_path):
    target = write(tmp_path, "big.md",
                   page(PROFILED) + "a" * (ACCEPTED["max_artifact_bytes"] + 1))
    assert target.stat().st_size > ACCEPTED["max_artifact_bytes"]
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R1"]


def test_r2_undecodable_bytes(tmp_path):
    target = write(tmp_path, "binary.md", b"---\ntype: analysis\n---\n\xff\xfe\n")
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R2"]


def test_only_r5_r6_and_r7_r8_collide(tmp_path):
    built = {row: shape(observe_page_okf(build(tmp_path))) for row, build in ROW_CASES}
    built["R0"] = shape(observe_page_okf(tmp_path / "absent.md"))
    built["R1"] = shape(observe_page_okf(
        write(tmp_path, "oversize.md",
              page(PROFILED) + "a" * (ACCEPTED["max_artifact_bytes"] + 1))))
    built["R2"] = shape(observe_page_okf(
        write(tmp_path, "undec.md", b"---\ntype: analysis\n---\n\xff\n")))
    groups: dict[tuple, list[str]] = {}
    for row, value in built.items():
        groups.setdefault(value, []).append(row)
    collisions = {tuple(sorted(rows)) for rows in groups.values() if len(rows) > 1}
    assert collisions == {("R5", "R6"), ("R7", "R8")}


def test_r0_r1_r2_r3_stay_four_distinct_values(tmp_path):
    values = {
        shape(observe_page_okf(tmp_path / "absent.md")),
        shape(observe_page_okf(write(tmp_path, "big.md",
                                     page(PROFILED) + "a" * (ACCEPTED["max_artifact_bytes"] + 1)))),
        shape(observe_page_okf(write(tmp_path, "bin.md", b"---\n\xff\n"))),
        shape(observe_page_okf(write(tmp_path, "legacy.md", "# Legacy\n"))),
    }
    assert len(values) == 4


# ---- accepted precedence constructions --------------------------------------

PRECEDENCE = [
    ("out-of-subset and unknown type", 'type: Made Up\ntags: ["a"]', "R10"),
    ("out-of-subset and unknown version",
     'type: analysis\nframework_profile: "9.9-rc.9"\ntags: ["a"]', "R10"),
    ("unknown type and unknown version",
     'type: "Made Up Type"\nframework_profile: "9.9-rc.9"', "R11"),
    ("duplicate key and missing type", "title: a\ntitle: b", "R8"),
    ("out-of-subset and missing type", 'title: ["a"]', "R9"),
    ("duplicate by spelling", "type: analysis\ntype: other", "R8"),
    ("duplicate by resolved identity 1 vs 01", "type: analysis\n1: a\n01: b", "R8"),
    ("duplicate by resolved identity yes vs true",
     "type: analysis\nyes: a\ntrue: b", "R8"),
    ("two merge keys in one mapping",
     'type: analysis\nbase: &b\n  k: "v"\n<<: *b\n<<: *b', "R8"),
    ("framework_profile with a non-scalar value",
     'type: analysis\nframework_profile:\n  nested: "0.1-rc.1"', "R12"),
]


@pytest.mark.parametrize("label,body,row", PRECEDENCE, ids=[p[0] for p in PRECEDENCE])
def test_precedence_lands_in_exactly_one_accepted_row(tmp_path, label, body, row):
    target = write(tmp_path, "case.md", page(body))
    expected = ROW_SHAPE[row]
    if row == "R10" and "9.9-rc.9" in body:
        expected = ROW_SHAPE["R10"]      # out-of-subset wins before version
    assert shape(observe_page_okf(target)) == expected


# ---- both sides of every declared numeric ceiling ----------------------------

def test_h01_total_artifact_bytes(tmp_path):
    base = page(PROFILED)
    pad = ACCEPTED["max_artifact_bytes"] - len(base.encode("utf-8"))
    at_max = write(tmp_path, "h01max.md", base + "a" * pad)
    over = write(tmp_path, "h01plus.md", base + "a" * (pad + 1))
    assert at_max.stat().st_size == ACCEPTED["max_artifact_bytes"]
    assert over.stat().st_size == ACCEPTED["max_artifact_bytes"] + 1
    assert shape(observe_page_okf(at_max)) == ROW_SHAPE["R14"]
    assert shape(observe_page_okf(over)) == ROW_SHAPE["R1"]


def test_h02_frontmatter_bytes(tmp_path):
    at_max_fm = frontmatter_of_bytes(ACCEPTED["max_frontmatter_bytes"])
    over_fm = frontmatter_of_bytes(ACCEPTED["max_frontmatter_bytes"] + 1)
    assert len(at_max_fm.encode("utf-8")) == ACCEPTED["max_frontmatter_bytes"]
    assert len(over_fm.encode("utf-8")) == ACCEPTED["max_frontmatter_bytes"] + 1
    assert shape(observe_page_okf(write(tmp_path, "a.md", page(at_max_fm)))) == ROW_SHAPE["R14"]
    assert shape(observe_page_okf(write(tmp_path, "b.md", page(over_fm)))) == ROW_SHAPE["R5"]


def test_h03_frontmatter_lines(tmp_path):
    at_max = PROFILED + "\n" + "\n".join(
        f'k{i}: "v"' for i in range(ACCEPTED["max_frontmatter_lines"] - 2))
    over = PROFILED + "\n" + "\n".join(
        f'k{i}: "v"' for i in range(ACCEPTED["max_frontmatter_lines"] - 1))
    assert at_max.count("\n") + 1 == ACCEPTED["max_frontmatter_lines"]
    assert over.count("\n") + 1 == ACCEPTED["max_frontmatter_lines"] + 1
    assert shape(observe_page_okf(write(tmp_path, "a.md", page(at_max)))) == ROW_SHAPE["R14"]
    assert shape(observe_page_okf(write(tmp_path, "b.md", page(over)))) == ROW_SHAPE["R5"]


def test_h04_single_line_length(tmp_path):
    at_max = PROFILED + "\n" + 'note: "' + "a" * (ACCEPTED["max_line_length"] - 8) + '"'
    over = PROFILED + "\n" + 'note: "' + "a" * (ACCEPTED["max_line_length"] - 7) + '"'
    assert max(len(x) for x in at_max.splitlines()) == ACCEPTED["max_line_length"]
    assert max(len(x) for x in over.splitlines()) == ACCEPTED["max_line_length"] + 1
    assert shape(observe_page_okf(write(tmp_path, "a.md", page(at_max)))) == ROW_SHAPE["R14"]
    assert shape(observe_page_okf(write(tmp_path, "b.md", page(over)))) == ROW_SHAPE["R5"]


def test_h05_exact_scanned_token_boundary(tmp_path):
    """The accepted exact pair: 9983 commas -> 10000 tokens -> R7; 9984 -> 10001 -> R6.

    A near-maximum proxy does not satisfy this obligation, so the token counts are
    measured here rather than assumed.
    """
    at_max = comma_block(9983)
    over = comma_block(9984)
    assert scanned_tokens(at_max) == ACCEPTED["max_tokens"]
    assert scanned_tokens(over) == ACCEPTED["max_tokens"] + 1
    at_max_obs = observe_page_okf(write(tmp_path, "a.md", page(at_max)))
    over_obs = observe_page_okf(write(tmp_path, "b.md", page(over)))
    assert shape(at_max_obs) == ROW_SHAPE["R7"]
    assert at_max_obs.okf_concept_reason != "OKF_PARSE_LIMIT_EXCEEDED"
    assert shape(over_obs) == ROW_SHAPE["R6"]
    assert over_obs.okf_concept_reason == "OKF_PARSE_LIMIT_EXCEEDED"


def test_h05_supplemental_cases(tmp_path):
    nearmax = chunked_flow(1988)
    order = chunked_flow(4993) + "\nbad:\n\ttabbed: 1"
    assert shape(observe_page_okf(write(tmp_path, "n.md", page(nearmax)))) == ROW_SHAPE["R10"]
    assert shape(observe_page_okf(write(tmp_path, "o.md", page(order)))) == ROW_SHAPE["R6"]


def test_h06_unique_nodes(tmp_path):
    assert shape(observe_page_okf(
        write(tmp_path, "a.md", page(chunked_flow(1989))))) == ROW_SHAPE["R10"]
    assert shape(observe_page_okf(
        write(tmp_path, "b.md", page(chunked_flow(1990))))) == ROW_SHAPE["R6"]


def test_h07_traversal_depth(tmp_path):
    assert shape(observe_page_okf(
        write(tmp_path, "a.md", page(nested_mapping(32))))) == ROW_SHAPE["R14"]
    assert shape(observe_page_okf(
        write(tmp_path, "b.md", page(nested_mapping(33))))) == ROW_SHAPE["R6"]


def test_h08_scalar_length(tmp_path):
    assert shape(observe_page_okf(
        write(tmp_path, "a.md", page(folded_scalar(ACCEPTED["max_scalar_length"]))))) == ROW_SHAPE["R14"]
    assert shape(observe_page_okf(
        write(tmp_path, "b.md", page(folded_scalar(ACCEPTED["max_scalar_length"] + 1))))) == ROW_SHAPE["R6"]


def test_h09_mapping_items(tmp_path):
    assert shape(observe_page_okf(
        write(tmp_path, "a.md", page(flow_mapping(ACCEPTED["max_mapping_items"]))))) == ROW_SHAPE["R10"]
    assert shape(observe_page_okf(
        write(tmp_path, "b.md", page(flow_mapping(ACCEPTED["max_mapping_items"] + 1))))) == ROW_SHAPE["R6"]


def test_h10_sequence_items(tmp_path):
    at_max = PROFILED + "\n" + _flow_sequence("pad", ['"v"'] * ACCEPTED["max_sequence_items"])
    over = PROFILED + "\n" + _flow_sequence("pad", ['"v"'] * (ACCEPTED["max_sequence_items"] + 1))
    assert shape(observe_page_okf(write(tmp_path, "a.md", page(at_max)))) == ROW_SHAPE["R10"]
    assert shape(observe_page_okf(write(tmp_path, "b.md", page(over)))) == ROW_SHAPE["R6"]


def test_h11_alias_count(tmp_path):
    assert shape(observe_page_okf(
        write(tmp_path, "a.md", page(aliases(ACCEPTED["max_aliases"]))))) == ROW_SHAPE["R10"]
    assert shape(observe_page_okf(
        write(tmp_path, "b.md", page(aliases(ACCEPTED["max_aliases"] + 1))))) == ROW_SHAPE["R6"]


def test_h12_alias_cycle(tmp_path):
    target = write(tmp_path, "cycle.md", page(PROFILED + "\ncycle: &c\n  self: *c"))
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R6"]


def test_h13_pathological_recursion(tmp_path):
    body = PROFILED + "\nx: " + "[" * 1500 + "]" * 1500
    assert shape(observe_page_okf(write(tmp_path, "rec.md", page(body)))) == ROW_SHAPE["R6"]


# ---- framing variants owned by this boundary --------------------------------

def test_crlf_framing_is_accepted(tmp_path):
    target = write(tmp_path, "crlf.md", page(PROFILED).replace("\n", "\r\n"))
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R14"]


def test_closing_delimiter_without_trailing_newline(tmp_path):
    target = write(tmp_path, "no_nl.md", "---\n" + PROFILED + "\n---")
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R14"]


def test_blank_line_before_opening_delimiter_is_legacy(tmp_path):
    target = write(tmp_path, "blank.md", "\n---\n" + PROFILED + "\n---\n")
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R3"]


def test_multiple_documents_in_frontmatter(tmp_path):
    target = write(tmp_path, "multi.md", page("type: analysis\n...\ntype: other"))
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R7"]


def test_empty_type_is_type_missing(tmp_path):
    assert shape(observe_page_okf(write(tmp_path, "e.md", page("type:")))) == ROW_SHAPE["R9"]


# ---- declared lexeme rules ---------------------------------------------------

def test_declared_lexeme_is_echoed_verbatim_for_unsupported_versions(tmp_path):
    for lexeme in ("9.9-rc.9", "0.1-RC.1", "0.1-rc.10", " 0.1-rc.1"):
        target = write(tmp_path, "v.md",
                       page(f'type: analysis\nframework_profile: "{lexeme}"'))
        result = observe_page_okf(target)
        assert result.framework_profile_reason == "PROFILE_VERSION_UNSUPPORTED"
        assert result.declared_framework_profile == lexeme


def test_declared_lexeme_absent_unless_a_version_was_read(tmp_path):
    for name, body, row in (
        ("legacy", "# Legacy\n", "R3"),
        ("no_profile", page('type: analysis\ntitle: "x"'), "R12"),
        ("unknown_type", page('type: "Made Up"\nframework_profile: "0.1-rc.1"'), "R11"),
        ("out_of_subset", page('type: analysis\nframework_profile: "0.1-rc.1"\nt: ["a"]'),
         "R10"),
    ):
        result = observe_page_okf(write(tmp_path, f"{name}.md", body))
        assert result.declared_framework_profile is None, name
        assert shape(result) == ROW_SHAPE[row], name


# ---- boundary hygiene --------------------------------------------------------

VOCABULARY = {
    "read_status": {"ok", "target_absent", "oversize", "undecodable"},
    "okf_concept_result": {"pass", "fail", "unverified", "not_evaluated"},
    "okf_concept_reason": {None, "OKF_FRONTMATTER_MISSING", "OKF_PARSE_LIMIT_EXCEEDED",
                           "OKF_TYPE_MISSING", "OKF_YAML_INVALID"},
    "framework_profile_result": {"pass", "fail", "not_applicable"},
    "framework_profile_reason": {None, "PROFILE_TYPE_UNSUPPORTED",
                                 "PROFILE_VERSION_UNSUPPORTED",
                                 "PROFILE_YAML_OUT_OF_SUBSET"},
    "execution_eligibility": {"not_evaluated"},
}

HOSTILE = [
    "---\ntype: analysis\n\tbad: 1\n---\n",
    "---\n" + "\ttab\n" * 50 + "---\n",
    "---\ntype: analysis\nsecret: SUPERSECRET-CANARY\n---\n",
    "---\n" + "[" * 900 + "]" * 900 + "\n---\n",
    "---\ntype: analysis\ncycle: &c\n  self: *c\n---\n",
]


@pytest.mark.parametrize("body", HOSTILE)
def test_hostile_input_is_bounded_deterministic_and_never_echoed(tmp_path, body):
    target = write(tmp_path, "hostile.md", body)
    first = observe_page_okf(target)
    assert first == observe_page_okf(target)          # deterministic
    for field, allowed in VOCABULARY.items():
        assert getattr(first, field) in allowed, field
    # The declared lexeme is the only free-form field; nothing else can carry
    # input, a traceback, or a path.
    rendered = repr(first)
    assert "Traceback" not in rendered
    assert "SUPERSECRET-CANARY" not in rendered
    assert str(tmp_path) not in rendered


def test_no_exception_escapes_for_any_accepted_row(tmp_path):
    for name, data in (
        ("absent.md", None),
        ("empty.md", b""),
        ("only_delimiter.md", b"---"),
        ("nul.md", b"---\ntype: analysis\n---\n\x00\xff\n"),
        ("bom.md", "﻿---\ntype: analysis\n---\n".encode("utf-8")),
    ):
        target = tmp_path / name
        if data is not None:
            target.write_bytes(data)
        result = observe_page_okf(target)
        assert isinstance(result, PageOkfObservation)
        assert result.execution_eligibility == "not_evaluated"


def test_observation_is_frozen(tmp_path):
    result = observe_page_okf(write(tmp_path, "v.md", page(PROFILED)))
    with pytest.raises(Exception):
        result.read_status = "ok"       # type: ignore[misc]


def test_accepts_str_and_path_inputs(tmp_path):
    target = write(tmp_path, "v.md", page(PROFILED))
    assert observe_page_okf(str(target)) == observe_page_okf(target)


# ---- M002 amendment A1: a present but unreadable regular target (R0U) --------

R0U_SHAPE = ("unreadable", "not_evaluated", None, "not_applicable", None,
             "not_evaluated", None)


def test_stat_permission_error_is_unreadable(tmp_path, monkeypatch):
    target = write(tmp_path, "denied.md", page(PROFILED))
    real_stat = Path.stat

    def denied(self, *args, **kwargs):
        if self.name == "denied.md":
            raise PermissionError(13, "permission denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    assert shape(observe_page_okf(target)) == R0U_SHAPE


def test_sharing_style_oserror_on_stat_is_unreadable(tmp_path, monkeypatch):
    target = write(tmp_path, "shared.md", page(PROFILED))
    real_stat = Path.stat

    def busy(self, *args, **kwargs):
        if self.name == "shared.md":
            raise OSError(32, "the process cannot access the file")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", busy)
    assert shape(observe_page_okf(target)) == R0U_SHAPE


def test_open_permission_error_after_regular_stat_is_unreadable(tmp_path, monkeypatch):
    target = write(tmp_path, "openfail.md", page(PROFILED))
    real_open = builtins.open

    def denied(path, *args, **kwargs):
        if str(path).endswith("openfail.md"):
            raise PermissionError(13, "permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", denied)
    assert shape(observe_page_okf(target)) == R0U_SHAPE


def test_read_oserror_after_open_is_unreadable(tmp_path, monkeypatch):
    target = write(tmp_path, "readfail.md", page(PROFILED))
    real_open = builtins.open

    class _Failing:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *args):
            raise OSError(5, "input/output error")

    def failing(path, *args, **kwargs):
        if str(path).endswith("readfail.md"):
            return _Failing()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing)
    assert shape(observe_page_okf(target)) == R0U_SHAPE


def test_absence_at_stat_stays_r0(tmp_path):
    assert shape(observe_page_okf(tmp_path / "never.md")) == ROW_SHAPE["R0"]


def test_target_disappearing_between_stat_and_open_stays_r0(tmp_path, monkeypatch):
    target = write(tmp_path, "racing.md", page(PROFILED))
    real_open = builtins.open

    def vanished(path, *args, **kwargs):
        if str(path).endswith("racing.md"):
            raise FileNotFoundError(2, "no such file")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", vanished)
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R0"]


def test_not_a_directory_error_stays_r0(tmp_path, monkeypatch):
    target = write(tmp_path, "notadir.md", page(PROFILED))
    real_stat = Path.stat

    def notadir(self, *args, **kwargs):
        if self.name == "notadir.md":
            raise NotADirectoryError(20, "not a directory")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", notadir)
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R0"]


def test_directory_target_stays_r0(tmp_path):
    directory = tmp_path / "dir.md"
    directory.mkdir()
    assert shape(observe_page_okf(directory)) == ROW_SHAPE["R0"]


def test_unreadable_result_leaks_no_path_or_diagnostic(tmp_path, monkeypatch):
    target = write(tmp_path, "secret-name.md", page(PROFILED))
    real_stat = Path.stat

    def denied(self, *args, **kwargs):
        if self.name == "secret-name.md":
            raise PermissionError(13, "SUPERSECRET-DIAGNOSTIC")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    rendered = repr(observe_page_okf(target))
    assert "SUPERSECRET-DIAGNOSTIC" not in rendered
    assert "secret-name" not in rendered
    assert str(tmp_path) not in rendered
    assert "Traceback" not in rendered


def test_r0u_is_distinct_from_every_other_read_status(tmp_path, monkeypatch):
    values = {
        shape(observe_page_okf(tmp_path / "absent.md")),
        shape(observe_page_okf(write(tmp_path, "big.md",
              page(PROFILED) + "a" * (ACCEPTED["max_artifact_bytes"] + 1)))),
        shape(observe_page_okf(write(tmp_path, "bin.md", b"---\n\xff\n"))),
        shape(observe_page_okf(write(tmp_path, "legacy.md", "# Legacy\n"))),
    }
    target = write(tmp_path, "denied.md", page(PROFILED))
    real_stat = Path.stat

    def denied(self, *args, **kwargs):
        if self.name == "denied.md":
            raise PermissionError(13, "denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    values.add(shape(observe_page_okf(target)))
    assert len(values) == 5


# ---- M002 amendment A3: the exact declared source lexeme ---------------------

BACKSLASH = chr(92)
LEXEME_CASES = [
    ("plain", "0.1-rc.1", "0.1-rc.1", True),
    ("double quoted", '"0.1-rc.1"', "0.1-rc.1", True),
    ("hex escape", '"0.1' + BACKSLASH + 'x2drc.1"',
     "0.1" + BACKSLASH + "x2drc.1", False),
    ("unicode escape", '"0.1' + BACKSLASH + 'u002drc.1"',
     "0.1" + BACKSLASH + "u002drc.1", False),
    ("hex escape, other version", '"9.9' + BACKSLASH + 'x2drc.9"',
     "9.9" + BACKSLASH + "x2drc.9", False),
    ("ordinary unsupported", '"9.9-rc.9"', "9.9-rc.9", False),
]


@pytest.mark.parametrize("label,source,expected,matches", LEXEME_CASES,
                         ids=[c[0] for c in LEXEME_CASES])
def test_declared_lexeme_is_the_exact_source_content(tmp_path, label, source,
                                                     expected, matches):
    target = write(tmp_path, "lex.md",
                   page("type: analysis\nframework_profile: " + source))
    result = observe_page_okf(target)
    assert result.declared_framework_profile == expected, label
    if matches:
        assert result.framework_profile_result == "pass"
        assert result.framework_profile_reason is None
    else:
        assert result.framework_profile_result == "fail"
        assert result.framework_profile_reason == "PROFILE_VERSION_UNSUPPORTED"


def test_escaped_spelling_is_not_accepted_as_the_candidate(tmp_path):
    """The defect this replaces: YAML decodes the escape to a hyphen, so the old
    implementation matched the candidate on a different source spelling."""
    target = write(tmp_path, "escaped.md",
                   page('type: analysis\nframework_profile: "0.1'
                        + BACKSLASH + 'x2drc.1"'))
    result = observe_page_okf(target)
    assert result.declared_framework_profile != "0.1-rc.1"
    assert result.framework_profile_result == "fail"


def test_lexeme_survives_surrounding_comment_and_whitespace(tmp_path):
    for name, body in (
        ("comment", "type: analysis\nframework_profile: 0.1-rc.1   # note"),
        ("pre-space", 'type: analysis\nframework_profile:    "0.1-rc.1"'),
        ("blank-lines", 'type: analysis\n\nframework_profile: "0.1-rc.1"\n'),
    ):
        result = observe_page_okf(write(tmp_path, name + ".md", page(body)))
        assert result.declared_framework_profile == "0.1-rc.1", name
        assert result.framework_profile_result == "pass", name


def test_single_quoted_and_block_scalars_stay_out_of_subset(tmp_path):
    for name, body in (
        ("single", "type: analysis\nframework_profile: '0.1-rc.1'"),
        ("block", "type: analysis\nframework_profile: >-\n  0.1-rc.1"),
    ):
        result = observe_page_okf(write(tmp_path, name + ".md", page(body)))
        assert result.framework_profile_reason == "PROFILE_YAML_OUT_OF_SUBSET", name
        assert result.declared_framework_profile is None, name


def test_no_lexeme_is_exposed_for_earlier_precedence_rows(tmp_path):
    for name, body in (
        ("R7 invalid", "type: analysis\n\tbad: 1"),
        ("R8 duplicate", 'type: analysis\ntype: other\nframework_profile: "0.1-rc.1"'),
        ("R9 no type", 'framework_profile: "0.1-rc.1"'),
        ("R10 out of subset",
         'type: analysis\nframework_profile: "0.1-rc.1"\ntags: ["a"]'),
        ("R11 unknown type", 'type: "Made Up"\nframework_profile: "0.1-rc.1"'),
        ("R12 no profile field", 'type: analysis\ntitle: "x"'),
    ):
        result = observe_page_okf(write(tmp_path, "row.md", page(body)))
        assert result.declared_framework_profile is None, name


# ---- M002 amendment A2: LF/CRLF framing and the raw byte ceiling -------------

CR = chr(13)
NOT_TERMINATORS = [
    ("bare CR", CR),
    ("vertical tab", "\x0b"),
    ("form feed", "\x0c"),
    ("next line", "\x85"),
    ("line separator", chr(0x2028)),
    ("paragraph separator", chr(0x2029)),
]


@pytest.mark.parametrize("label,separator", NOT_TERMINATORS,
                         ids=[c[0] for c in NOT_TERMINATORS])
def test_only_lf_and_crlf_can_form_a_delimiter(tmp_path, label, separator):
    """A document 'framed' with any other separator has no frontmatter at all."""
    body = ("---" + separator + "type: analysis" + separator
            + 'framework_profile: "0.1-rc.1"' + separator + "---" + separator)
    assert shape(observe_page_okf(write(tmp_path, "sep.md", body))) == ROW_SHAPE["R3"]


def test_lf_crlf_and_mixed_endings_all_frame(tmp_path):
    lf = page(PROFILED)
    for name, text in (
        ("lf", lf),
        ("crlf", lf.replace("\n", "\r\n")),
        ("mixed", '---\ntype: analysis\r\nframework_profile: "0.1-rc.1"\n---\n'),
        ("no trailing newline",
         '---\ntype: analysis\nframework_profile: "0.1-rc.1"\n---'),
        ("crlf no trailing newline",
         '---\r\ntype: analysis\r\nframework_profile: "0.1-rc.1"\r\n---'),
    ):
        assert shape(observe_page_okf(write(tmp_path, name + ".md", text))) == \
            ROW_SHAPE["R14"], name


def test_whitespace_around_a_delimiter_is_not_a_delimiter(tmp_path):
    for name, text in (
        ("leading space", " ---\ntype: analysis\n---\n"),
        ("trailing space", "--- \ntype: analysis\n---\n"),
        ("blank first line", "\n---\ntype: analysis\n---\n"),
    ):
        result = observe_page_okf(write(tmp_path, name + ".md", text))
        assert result.okf_concept_result == "not_evaluated", name


def raw_block(target_bytes: int, ending: str) -> str:
    """A frontmatter block of exactly ``target_bytes`` raw UTF-8 bytes."""
    lines = ["type: analysis", 'framework_profile: "0.1-rc.1"']

    def size(candidate):
        return len(ending.join(candidate).encode("utf-8"))

    index = 0
    while size(lines + ['k%03d: "%s"' % (index, "a" * 300)]) + 40 < target_bytes:
        lines.append('k%03d: "%s"' % (index, "a" * 300))
        index += 1
    head = 'tail: "'
    fill = target_bytes - size(lines) - len(ending.encode("utf-8")) - len(head) - 1
    lines.append(head + "a" * fill + '"')
    assert size(lines) == target_bytes
    return ending.join(lines)


@pytest.mark.parametrize("ending_name,ending", [("lf", "\n"), ("crlf", "\r\n")])
def test_raw_frontmatter_byte_ceiling_counts_source_bytes(tmp_path, ending_name,
                                                          ending):
    ceiling = ACCEPTED["max_frontmatter_bytes"]
    at_max = raw_block(ceiling, ending)
    over = raw_block(ceiling + 1, ending)
    assert len(at_max.encode("utf-8")) == ceiling
    assert len(over.encode("utf-8")) == ceiling + 1
    for name, block, expected in (("max", at_max, "R14"), ("plus1", over, "R5")):
        text = "---" + ending + block + ending + "---" + ending
        assert shape(observe_page_okf(
            write(tmp_path, ending_name + "_" + name + ".md", text))) == \
            ROW_SHAPE[expected], ending_name + " " + name


def test_crlf_block_is_not_measured_by_its_normalized_length(tmp_path):
    """The reviewer's case: a block whose LF-normalized length is exactly at the
    ceiling but whose real CRLF bytes exceed it must refuse."""
    ceiling = ACCEPTED["max_frontmatter_bytes"]
    lines = ["type: analysis", 'framework_profile: "0.1-rc.1"']
    index = 0
    while len("\n".join(lines + ['k%03d: "%s"' % (index, "a" * 300)])
              .encode("utf-8")) + 40 < ceiling:
        lines.append('k%03d: "%s"' % (index, "a" * 300))
        index += 1
    head = 'tail: "'
    fill = ceiling - len("\n".join(lines).encode("utf-8")) - 1 - len(head) - 1
    lines.append(head + "a" * fill + '"')
    block = "\r\n".join(lines)
    assert len(block.replace("\r\n", "\n").encode("utf-8")) == ceiling
    assert len(block.encode("utf-8")) > ceiling
    text = "---\r\n" + block + "\r\n---\r\n"
    assert shape(observe_page_okf(write(tmp_path, "normalized.md", text))) == \
        ROW_SHAPE["R5"]


def test_line_count_and_length_use_accepted_endings(tmp_path):
    lines_ceiling = ACCEPTED["max_frontmatter_lines"]
    body = ["type: analysis", 'framework_profile: "0.1-rc.1"']
    body += ['k%03d: "v"' % i for i in range(lines_ceiling - 2)]
    at_max = "\r\n".join(body)
    over = "\r\n".join(body + ['extra: "v"'])
    assert shape(observe_page_okf(write(tmp_path, "lmax.md",
                 "---\r\n" + at_max + "\r\n---\r\n"))) == ROW_SHAPE["R14"]
    assert shape(observe_page_okf(write(tmp_path, "lover.md",
                 "---\r\n" + over + "\r\n---\r\n"))) == ROW_SHAPE["R5"]

    length_ceiling = ACCEPTED["max_line_length"]
    long_ok = 'note: "' + "a" * (length_ceiling - 8) + '"'
    long_over = 'note: "' + "a" * (length_ceiling - 7) + '"'
    assert len(long_ok) == length_ceiling
    profiled_crlf = PROFILED.replace("\n", "\r\n")
    assert shape(observe_page_okf(write(tmp_path, "wmax.md",
                 "---\r\n" + profiled_crlf + "\r\n" + long_ok
                 + "\r\n---\r\n"))) == ROW_SHAPE["R14"]
    assert shape(observe_page_okf(write(tmp_path, "wover.md",
                 "---\r\n" + profiled_crlf + "\r\n" + long_over
                 + "\r\n---\r\n"))) == ROW_SHAPE["R5"]


def test_raw_block_is_bounded_before_any_yaml_parse(tmp_path, monkeypatch):
    """A block over the byte ceiling must refuse without PyYAML seeing it."""
    calls = []
    real_scan = yaml.scan

    def watched(*args, **kwargs):
        calls.append(1)
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(yaml, "scan", watched)
    over = raw_block(ACCEPTED["max_frontmatter_bytes"] + 1, "\n")
    result = observe_page_okf(write(tmp_path, "over.md",
                                    "---\n" + over + "\n---\n"))
    assert result.okf_concept_reason == "OKF_PARSE_LIMIT_EXCEEDED"
    assert calls == []


# ---- M002 addendum B1: an unusable path spelling is a non-resolving target ----


def test_embedded_nul_path_returns_r0(tmp_path):
    """Reproduced by Review Prompt 003: this raised ValueError from Path.stat()."""
    assert shape(observe_page_okf("embedded" + chr(0) + "nul")) == ROW_SHAPE["R0"]
    assert shape(observe_page_okf(str(tmp_path) + "/a" + chr(0) + "b.md")) == \
        ROW_SHAPE["R0"]


def test_overlong_path_component_returns_r0(tmp_path):
    """Reproduced by Review Prompt 003 as 'path too long for Windows'.

    Platforms that accept the spelling simply report an absent target, which is
    the same row, so the assertion holds either way.
    """
    assert shape(observe_page_okf("x" * 40_000)) == ROW_SHAPE["R0"]
    assert shape(observe_page_okf(str(tmp_path) + "/" + "y" * 40_000)) == \
        ROW_SHAPE["R0"]


def test_injected_filesystem_valueerror_maps_to_r0(tmp_path, monkeypatch):
    """Platform-independent proof of the mapping itself, not of one OS message."""
    target = write(tmp_path, "spelling.md", page(PROFILED))
    real_stat = Path.stat

    def unusable(self, *args, **kwargs):
        if self.name == "spelling.md":
            raise ValueError("stat: embedded null character in path")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unusable)
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R0"]


def test_injected_open_valueerror_maps_to_r0(tmp_path, monkeypatch):
    target = write(tmp_path, "openspell.md", page(PROFILED))
    real_open = builtins.open

    def unusable(path, *args, **kwargs):
        if str(path).endswith("openspell.md"):
            raise ValueError("embedded null byte")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", unusable)
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R0"]


def test_invalid_spelling_result_leaks_no_diagnostic(tmp_path, monkeypatch):
    target = write(tmp_path, "leaky.md", page(PROFILED))
    real_stat = Path.stat

    def unusable(self, *args, **kwargs):
        if self.name == "leaky.md":
            raise ValueError("stat: SUPERSECRET-SPELLING-DIAGNOSTIC")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unusable)
    rendered = repr(observe_page_okf(target))
    assert "SUPERSECRET-SPELLING-DIAGNOSTIC" not in rendered
    assert "leaky" not in rendered
    assert str(tmp_path) not in rendered
    assert "Traceback" not in rendered


def test_programming_errors_are_not_swallowed(tmp_path, monkeypatch):
    """The mapping is narrow: only the filesystem-boundary ValueError."""
    target = write(tmp_path, "boom.md", page(PROFILED))
    real_stat = Path.stat

    for exception in (MemoryError("out of memory"), KeyboardInterrupt()):
        def raising(self, *args, _exc=exception, **kwargs):
            if self.name == "boom.md":
                raise _exc
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", raising)
        with pytest.raises(type(exception)):
            observe_page_okf(target)


def test_a_well_behaved_pathlike_still_works(tmp_path):
    class _Like:
        def __init__(self, path):
            self._path = str(path)

        def __fspath__(self):
            return self._path

    target = write(tmp_path, "fspath.md", page(PROFILED))
    assert shape(observe_page_okf(_Like(target))) == ROW_SHAPE["R14"]
    assert shape(observe_page_okf(str(target))) == ROW_SHAPE["R14"]
    assert shape(observe_page_okf(Path(target))) == ROW_SHAPE["R14"]


# ---- M002 finding 2: the line parser must use bounded auxiliary memory --------


def test_a_legacy_input_consumes_only_its_first_line(monkeypatch):
    """Structural laziness: framing returns after one line, without collecting."""
    real_iter = obs._iter_lines
    consumed = []

    def counting(text):
        for item in real_iter(text):
            consumed.append(1)
            yield item

    monkeypatch.setattr(obs, "_iter_lines", counting)
    block = obs._frame_frontmatter("# Legacy\n" + "x\n" * 200_000)
    assert block.status == "legacy"
    assert len(consumed) == 1


def test_framing_never_materializes_the_whole_line_stream(monkeypatch):
    """No list/tuple of all lines is built: the iterator is consumed lazily and the
    parser keeps only counters and offsets."""
    real_iter = obs._iter_lines
    live = {"max_outstanding": 0}

    def watching(text):
        for index, item in enumerate(real_iter(text), start=1):
            live["max_outstanding"] = index
            yield item

    monkeypatch.setattr(obs, "_iter_lines", watching)
    block = obs._frame_frontmatter("---\n" + "k: v\n" * 200_000 + "---\n")
    assert block.status == "ok"
    assert block.over_bound is True
    assert block.raw == ""              # nothing was copied out of a hostile block
    assert block.line_count == 200_000


def test_near_maximum_newline_heavy_inputs_stay_within_bounded_memory(tmp_path):
    """Reviewer-equivalent tracemalloc evidence for the two 1 MiB cases.

    Review Prompt 003 measured 172.0 MiB and 189.1 MiB peak traced allocation.
    The threshold here is evidence of bounded auxiliary memory, not a timing claim.
    """
    import tracemalloc

    ceiling = ACCEPTED["max_artifact_bytes"]
    cases = {
        "legacy, only LF": "\n" * ceiling,
        "framed, blank content lines": "---\n" + "\n" * (ceiling - 8) + "---\n",
    }
    for label, text in cases.items():
        tracemalloc.start()
        block = obs._frame_frontmatter(text)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < 16 * 1024 * 1024, f"{label}: {peak / 1024 / 1024:.1f} MiB"
        assert block.status in ("legacy", "ok")


def test_over_bound_terminated_and_unterminated_keep_r5_r4_precedence(tmp_path):
    """Framing still precedes bounds: a closing delimiter makes an over-bound block
    R5, and its absence makes the same content R4."""
    body = "k: v\n" * (ACCEPTED["max_frontmatter_lines"] + 100)
    terminated = write(tmp_path, "term.md", "---\n" + body + "---\n")
    unterminated = write(tmp_path, "unterm.md", "---\n" + body)
    assert shape(observe_page_okf(terminated)) == ROW_SHAPE["R5"]
    assert shape(observe_page_okf(unterminated)) == ROW_SHAPE["R4"]


def test_hundreds_of_thousands_of_short_lines_classify_correctly(tmp_path):
    # 150,000 five-byte lines stay under the 1,048,576-byte total ceiling, so the
    # frontmatter bounds decide the row rather than R1.
    body = "k: v\n" * 150_000
    terminated = write(tmp_path, "many_term.md", "---\n" + body + "---\n")
    unterminated = write(tmp_path, "many_unterm.md", "---\n" + body)
    assert terminated.stat().st_size < ACCEPTED["max_artifact_bytes"]
    assert shape(observe_page_okf(terminated)) == ROW_SHAPE["R5"]
    assert shape(observe_page_okf(unterminated)) == ROW_SHAPE["R4"]


def test_a_line_heavy_page_over_the_total_ceiling_is_still_r1(tmp_path):
    """The outer ceiling still wins: it is checked before decode, so a 1.5 MB page
    never reaches framing at all."""
    body = "k: v\n" * 300_000
    target = write(tmp_path, "huge.md", "---\n" + body + "---\n")
    assert target.stat().st_size > ACCEPTED["max_artifact_bytes"]
    assert shape(observe_page_okf(target)) == ROW_SHAPE["R1"]


def test_an_over_bound_block_never_reaches_pyyaml(tmp_path, monkeypatch):
    calls = []
    real_scan = yaml.scan

    def watched(*args, **kwargs):
        calls.append(1)
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(yaml, "scan", watched)
    body = "k: v\n" * (ACCEPTED["max_frontmatter_lines"] + 10)
    observe_page_okf(write(tmp_path, "big.md", "---\n" + body + "---\n"))
    assert calls == []
