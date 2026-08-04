"""Skip-gated real-dependency smoke for live C# structure extraction.

Optional Slice 2 of
`02_analysis/tree_sitter_structured_compatibility_milestone.md`. The
preceding compatibility cleanup is covered by fake-parser unit tests
in `test_tree_sitter_language_compat.py`; those tests prove the
`_bind_tree_sitter_language` candidate sequence handles a
PyCapsule-shaped stand-in, but they cannot detect real-world API
drift in `tree-sitter` or `tree-sitter-c-sharp`.

This smoke runs the **live** Python tree-sitter binding plus the
real C# grammar package and asserts:

- the helper successfully binds the grammar (no shim was needed);
- the Unity bridge v1 classification (``kind == "unity_component"``
  on a direct `MonoBehaviour` subclass) survives the real walker;
- the framework-tagging slice's
  ``tags == ("framework:unity", "role:component")`` pair is
  attached to the same class;
- a nested method stays ``kind == "method"`` with ``tags == ()``.

When the optional `llloom[structured]` dependency stack is not
present, every test in this module skips cleanly via
``pytest.importorskip``. The default base suite passes unchanged.
"""

from __future__ import annotations

import hashlib

import pytest

# Slice 082a follow-up: skip-gates live INSIDE the test body so
# pytest collects exactly one item and reports `1 skipped` (exit
# code 0) when the optional extras are absent. The prior
# module-level pattern produced `collected 0 items / 1 skipped`
# which exits 5 (`NO_TESTS_COLLECTED`) under stricter pytest
# configurations. Behavior when the optional extra IS installed
# is byte-identical to the pre-082a Slice 065 smoke.


_FIXTURE_CSHARP = (
    "using UnityEngine;\n"
    "\n"
    "public class PlayerController : MonoBehaviour\n"
    "{\n"
    "    void Start() {}\n"
    "}\n"
)


def _content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_real_csharp_grammar_extracts_unity_component_and_method() -> None:
    """End-to-end: real tree-sitter + real C# grammar + the package's
    `_bind_tree_sitter_language` shim. Confirms the live PyCapsule
    binding path works and that the Unity bridge + framework-tagging
    behaviour from the prior slices is intact.

    Slice 082a follow-up: skip-gates live inside the test body
    rather than at module scope so pytest collects exactly one
    item and reports `1 skipped` (exit code 0) when the optional
    extras are absent.
    """
    pytest.importorskip(
        "tree_sitter",
        reason='real-dependency smoke; install with: pip install "llloom[structured]"',
    )
    pytest.importorskip(
        "tree_sitter_c_sharp",
        reason='real-dependency smoke; install with: pip install "llloom[structured]"',
    )

    from llloom.structured.extract import extract_structure

    report = extract_structure(
        _FIXTURE_CSHARP,
        source_id="smoke.unity_component",
        source_class="code",
        locator_type="code_v1",
        raw_path="raw/sources/PlayerController.cs",
        content_hash=_content_hash(_FIXTURE_CSHARP),
    )

    assert report.language == "csharp"
    assert report.source_class == "code"
    assert report.locator_type == "code_v1"

    # The Unity component classification: direct MonoBehaviour
    # subclass → kind == "unity_component" plus framework tags.
    by_path = {item.symbol_path: item for item in report.items}
    assert "PlayerController" in by_path, (
        f"expected a top-level PlayerController item; got "
        f"{sorted(by_path)}"
    )
    cls = by_path["PlayerController"]
    assert cls.kind == "unity_component", (
        f"direct MonoBehaviour subclass must carry kind=unity_component; "
        f"got {cls.kind!r}"
    )
    assert cls.tags == ("framework:unity", "role:component"), (
        f"Unity bridge must attach (framework:unity, role:component); "
        f"got {cls.tags!r}"
    )

    # The nested Start() method must NOT inherit the class's
    # framework tags. The walker emits its symbol_path as
    # "PlayerController.Start" with kind=method and tags=().
    nested = [
        item for item in report.items
        if item.symbol_path.startswith("PlayerController.")
        and item.kind == "method"
    ]
    assert nested, (
        f"expected at least one nested method under PlayerController; "
        f"got items {[(i.symbol_path, i.kind) for i in report.items]}"
    )
    start_methods = [n for n in nested if n.symbol_path.endswith(".Start")]
    assert start_methods, (
        f"expected a PlayerController.Start method item; got "
        f"{[(n.symbol_path, n.kind) for n in nested]}"
    )
    start = start_methods[0]
    assert start.kind == "method"
    assert start.tags == (), (
        f"nested methods must not inherit class framework tags; "
        f"got {start.tags!r}"
    )
