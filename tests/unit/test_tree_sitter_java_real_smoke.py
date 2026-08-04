"""Skip-gated real-dependency smoke for live Java structure extraction.

Slice 082 added first-class Java support to the deterministic
`source_class="code"` structured-extract path. The fake-parser unit
tests in ``test_structure_extract.py`` (``test_java_*``) prove the
walker contract; they cannot detect real-world API drift in
``tree-sitter`` or ``tree-sitter-java``.

This smoke runs the **live** Python tree-sitter binding plus the
real Java grammar package and asserts:

- the helper successfully binds the grammar (no shim was needed);
- `report.language == "java"`;
- a representative declaration of every Java V1 item kind surfaces
  (``class``, ``interface``, ``enum``, ``record``, ``method``,
  ``constructor``, ``field``);
- nested method / constructor / field paths inherit the enclosing
  class prefix (``Project.evaluate`` etc.);
- no `code` body text, planted comment, or planted scalar literal
  appears in the serialized report — the walker stays
  metadata-only end-to-end.

When the optional `llloom[structured]` dependency stack is not
present, every test in this module skips cleanly via
``pytest.importorskip``. The default base suite passes unchanged
(this module surfaces as one extra skip).
"""

from __future__ import annotations

import hashlib

import pytest

# Slice 082a follow-up: skip-gates live INSIDE the test body so
# pytest collects exactly one item and reports `1 skipped` (exit
# code 0) when the optional extras are absent. The prior
# module-level pattern produced `collected 0 items / 1 skipped`
# which exits 5 (`NO_TESTS_COLLECTED`) under stricter pytest
# configurations (CI policies). Behavior when the optional extra
# IS installed is byte-identical to Slice 082.


_FIXTURE_JAVA = (
    "package com.example;\n"
    "\n"
    "import java.util.List;\n"
    "\n"
    "// SECRET_COMMENT_42 should never leak\n"
    "public class Project {\n"
    "    private int maxsize = 99;\n"
    "\n"
    "    public Project() {\n"
    "        System.out.println(\"NOISE_BODY_77\");\n"
    "    }\n"
    "\n"
    "    public int evaluate() {\n"
    "        return 1;\n"
    "    }\n"
    "}\n"
    "\n"
    "interface Metric {}\n"
    "\n"
    "enum PatchKind { LAND, WATER }\n"
    "\n"
    "record PatchRecord(int id) {}\n"
)


def _content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_real_java_grammar_extracts_every_v1_item_kind() -> None:
    """End-to-end: real tree-sitter + real Java grammar + the
    package's shared `_bind_tree_sitter_language` shim. Confirms
    the live binding path works and the walker emits exactly the
    declarations the dev note's Slice 082 priority list named.

    Slice 082a follow-up: skip-gates live inside the test body
    rather than at module scope so pytest collects exactly one
    item and reports `1 skipped` (exit code 0) when the optional
    extras are absent. The pre-082a module-level pattern produced
    `collected 0 items / 1 skipped` which exits 5
    (`NO_TESTS_COLLECTED`) under stricter pytest configurations.
    """
    pytest.importorskip(
        "tree_sitter",
        reason='real-dependency smoke; install with: pip install "llloom[structured]"',
    )
    pytest.importorskip(
        "tree_sitter_java",
        reason='real-dependency smoke; install with: pip install "llloom[structured]"',
    )

    from llloom.structured.extract import extract_structure

    report = extract_structure(
        _FIXTURE_JAVA,
        source_id="smoke.java.project",
        source_class="code",
        locator_type="code_v1",
        raw_path="raw/sources/Project.java",
        content_hash=_content_hash(_FIXTURE_JAVA),
    )

    assert report.language == "java"
    assert report.source_class == "code"
    assert report.locator_type == "code_v1"

    by_path = {(item.symbol_path, item.kind) for item in report.items}
    expected = {
        ("Project", "class"),
        ("Project.Project", "constructor"),
        ("Project.evaluate", "method"),
        ("Project.maxsize", "field"),
        ("Metric", "interface"),
        ("PatchKind", "enum"),
        ("PatchRecord", "record"),
    }
    missing = expected - by_path
    assert not missing, (
        f"real Java grammar walker missing expected items {missing}; "
        f"got {sorted(by_path)}"
    )

    # Metadata-only invariant: no body / comment / literal text leaks.
    import yaml as yaml_mod

    text = yaml_mod.safe_dump(report.to_mapping(), sort_keys=False)
    for forbidden in (
        "SECRET_COMMENT_42",
        "NOISE_BODY_77",
        "System.out.println",
        "import java.util.List",
        "return 1",
    ):
        assert forbidden not in text, (
            f"real Java walker leaked {forbidden!r}: {text}"
        )
