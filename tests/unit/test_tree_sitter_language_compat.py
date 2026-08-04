"""Unit tests for the tree-sitter language-binding compatibility shim.

Pinned by `02_analysis/tree_sitter_structured_compatibility_milestone.md`.
The bug: current `tree_sitter_c_sharp.language()` returns a raw
PyCapsule, and current `tree-sitter` rejects the capsule when bound
directly to a parser. The shared `_bind_tree_sitter_language` helper
must wrap the capsule with `tree_sitter.Language(...)` before
binding, while still accepting already-`Language` objects and the
older `parser.set_language(...)` API.

These tests inject a fake `tree_sitter` module into `sys.modules` so
the default suite does not require the optional
`llloom[structured]` extra.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


# A sentinel marker we use to distinguish the wrapped path. Modeled
# as a small dataclass-like wrapper around an arbitrary grammar
# payload (capsule, string, anything) so `isinstance(...,
# FakeLanguage)` is the marker the binder consults.
class _FakeLanguage:
    """Stand-in for `tree_sitter.Language`."""

    def __init__(self, grammar: Any) -> None:
        self.grammar = grammar

    def __repr__(self) -> str:  # pragma: no cover — debug aid only
        return f"_FakeLanguage(grammar={self.grammar!r})"


@pytest.fixture
def fake_tree_sitter(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake `tree_sitter` module exposing `Language`.

    The real `tree_sitter` package may or may not be installed in
    the local venv. We deliberately replace `sys.modules["tree_sitter"]`
    for the test so the helper picks up our fake class and so any
    assertions about isinstance(...) line up with the fake.
    """
    module = types.ModuleType("tree_sitter")
    module.Language = _FakeLanguage  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tree_sitter", module)
    yield module


class _FakeParserDirect:
    """Parser that accepts language objects via attribute assignment.

    Models the tree-sitter 0.22+ API. Rejects anything that is not a
    `_FakeLanguage`, matching the bug shape (current `tree-sitter`
    refuses a raw PyCapsule).
    """

    def __init__(self) -> None:
        # Bypass our own __setattr__ guard during construction.
        object.__setattr__(self, "language", None)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "language":
            if not isinstance(value, _FakeLanguage):
                raise TypeError(
                    f"parser.language requires a tree_sitter.Language, "
                    f"got {type(value).__name__}"
                )
        object.__setattr__(self, name, value)


class _FakeParserSetter:
    """Parser that exposes the legacy `set_language(...)` API only.

    Models tree-sitter 0.21-style bindings. Has no attribute slot
    for `language`; the helper must fall back to `set_language`.
    """

    def __init__(self) -> None:
        self._lang: Any = None
        # No `language` attribute exists, but assigning one would still
        # work via __dict__; emulate the older API by raising on direct
        # assignment.

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "language":
            raise AttributeError("can't set attribute 'language'")
        object.__setattr__(self, name, value)

    def set_language(self, language: Any) -> None:
        if not isinstance(language, _FakeLanguage):
            raise TypeError(
                f"set_language requires a tree_sitter.Language, "
                f"got {type(language).__name__}"
            )
        self._lang = language


class _FakeParserRejecting:
    """Parser that refuses every binding path.

    Used to confirm the helper raises a wrapped `StructureExtractError`
    instead of leaking a low-level exception.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "language":
            raise TypeError("rejecting parser refuses attribute binding")
        object.__setattr__(self, name, value)

    def set_language(self, language: Any) -> None:
        raise TypeError("rejecting parser refuses setter binding")


def test_bind_already_language_assigns_directly(
    fake_tree_sitter: types.ModuleType,
) -> None:
    """If the caller already passes a `tree_sitter.Language`, the
    helper assigns it directly without re-wrapping."""
    from llloom.structured.extract import _bind_tree_sitter_language

    parser = _FakeParserDirect()
    grammar = _FakeLanguage(grammar="csharp-grammar-payload")

    bound = _bind_tree_sitter_language(parser, grammar)

    assert bound is grammar, "already-Language should be assigned as-is"
    assert parser.language is grammar


def test_bind_pycapsule_like_grammar_is_wrapped(
    fake_tree_sitter: types.ModuleType,
) -> None:
    """A raw PyCapsule-like grammar (anything not a `Language`) is
    wrapped through `Language(grammar)` and the wrapped object is what
    the parser ends up bound to. This is the regression for the
    Vampire-main / current `tree_sitter_c_sharp` failure."""
    from llloom.structured.extract import _bind_tree_sitter_language

    parser = _FakeParserDirect()
    raw_capsule = object()  # stand-in for the PyCapsule

    bound = _bind_tree_sitter_language(parser, raw_capsule)

    assert isinstance(bound, _FakeLanguage)
    assert bound.grammar is raw_capsule
    assert parser.language is bound


def test_bind_pycapsule_falls_back_to_set_language(
    fake_tree_sitter: types.ModuleType,
) -> None:
    """If the parser only exposes the legacy `set_language(...)` API,
    the helper still finds it after wrapping the capsule."""
    from llloom.structured.extract import _bind_tree_sitter_language

    parser = _FakeParserSetter()
    raw_capsule = object()

    bound = _bind_tree_sitter_language(parser, raw_capsule)

    assert isinstance(bound, _FakeLanguage)
    assert bound.grammar is raw_capsule
    assert parser._lang is bound


def test_bind_direct_language_falls_back_to_set_language(
    fake_tree_sitter: types.ModuleType,
) -> None:
    """An already-`Language` object on an older `set_language`-only
    parser still binds via the legacy setter (no needless wrapping)."""
    from llloom.structured.extract import _bind_tree_sitter_language

    parser = _FakeParserSetter()
    grammar = _FakeLanguage(grammar="legacy-payload")

    bound = _bind_tree_sitter_language(parser, grammar)

    assert bound is grammar
    assert parser._lang is grammar


def test_bind_raises_structure_extract_error_when_all_paths_fail(
    fake_tree_sitter: types.ModuleType,
) -> None:
    """Every candidate refused → the helper surfaces a single
    `StructureExtractError` with the `llloom[structured]` install
    hint and a digest of the failed attempts."""
    from llloom.structured.extract import (
        StructureExtractError,
        _bind_tree_sitter_language,
    )

    parser = _FakeParserRejecting()
    raw_capsule = object()

    with pytest.raises(StructureExtractError) as exc_info:
        _bind_tree_sitter_language(parser, raw_capsule)
    msg = str(exc_info.value)
    assert "llloom[structured]" in msg
    # Mentions at least one underlying refusal so the user can grep.
    assert "rejecting parser" in msg


def test_build_parser_routes_through_compatibility_helper(
    fake_tree_sitter: types.ModuleType,
) -> None:
    """The shared `_build_parser(parser_cls, grammar)` entry point
    (used by every per-language loader) wraps capsule-like grammars
    via the helper before returning the bound parser. This is the
    one-shared-path guarantee from the milestone direction note."""
    from llloom.structured.extract import _build_parser

    raw_capsule = object()
    parser, bound = _build_parser(_FakeParserDirect, raw_capsule)

    assert isinstance(parser, _FakeParserDirect)
    assert isinstance(bound, _FakeLanguage)
    assert bound.grammar is raw_capsule
    assert parser.language is bound


def test_build_parser_passes_through_already_language(
    fake_tree_sitter: types.ModuleType,
) -> None:
    """A pre-built `Language` object is preserved unchanged by
    `_build_parser`."""
    from llloom.structured.extract import _build_parser

    grammar = _FakeLanguage(grammar="ts-payload")
    parser, bound = _build_parser(_FakeParserDirect, grammar)

    assert bound is grammar
    assert parser.language is grammar


def test_base_import_does_not_require_tree_sitter() -> None:
    """Regression: importing `llloom.structured.extract` must not
    require the optional `tree_sitter` package. The lazy import lives
    inside `_bind_tree_sitter_language`. We confirm by checking that
    the module imports cleanly with no `tree_sitter*` module in
    `sys.modules` before, and that nothing in `sys.modules` after the
    import points at a real `tree_sitter` package (the fake fixture
    is not active in this test)."""
    # The full default suite already runs with no `docling` / no
    # `tree_sitter` in the venv; this assertion only catches the case
    # where someone moves the lazy import to module top.
    import importlib

    pre = set(sys.modules)
    importlib.import_module("llloom.structured.extract")
    # If `tree_sitter` was already present in the venv we cannot assert
    # absence after; what we can assert is that importing the module
    # did not pull it in.
    if "tree_sitter" not in pre:
        assert "tree_sitter" not in sys.modules, (
            "importing llloom.structured.extract eagerly imported tree_sitter"
        )
