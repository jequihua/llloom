"""Unit tests for the deterministic structured-source extractor.

YAML extraction runs in the base install via PyYAML's ``compose``
API. Python extraction lazy-imports tree-sitter; these tests do
**not** require the real tree-sitter SDK — they use a meta-path
``ImportError`` blocker to prove the missing-extra refusal path.
"""

from __future__ import annotations

import sys
import types

import pytest

from llloom.structured import (
    StructureExtractError,
    extract_structure,
    write_structure_report,
)


_YAML_SOURCE = (
    "policies:\n"
    "  markdown_prose: claim_extract_and_view_render\n"
    "  legal_act: claim_extract\n"
    "defaults:\n"
    "  unknown: deny\n"
)


def _args(**overrides):
    base = dict(
        source_id="src.policies",
        source_class="structured_yaml",
        locator_type="code_v1",
        raw_path="raw/sources/policies.yaml",
        content_hash="sha256:deadbeef",
    )
    base.update(overrides)
    return base


# --- YAML: key paths + code_v1 locators ---------------------------------


def test_yaml_emits_mapping_key_paths() -> None:
    report = extract_structure(_YAML_SOURCE, **_args())
    assert report.language == "yaml"
    assert report.version == "structure_report_v1"
    symbol_paths = {item.symbol_path for item in report.items}
    assert "policies" in symbol_paths
    assert "policies.markdown_prose" in symbol_paths
    assert "policies.legal_act" in symbol_paths
    assert "defaults" in symbol_paths
    assert "defaults.unknown" in symbol_paths
    for item in report.items:
        assert item.kind == "mapping_key"
        assert item.locator["locator_type"] == "code_v1"
        assert item.locator["path"] == "raw/sources/policies.yaml"
        assert isinstance(item.locator["start_line"], int)
        assert item.locator["start_line"] >= 1


def test_yaml_omits_scalar_values_comments_and_bodies() -> None:
    source = (
        "# a leading comment about secrets\n"
        "policies:\n"
        "  secret_key: SECRET_VALUE_abc123\n"
        "  another: ANOTHER_SECRET\n"
        "# trailing comment\n"
    )
    report = extract_structure(source, **_args())
    # Neither the comments nor any scalar value should appear
    # anywhere in the serialized report.
    import yaml as yaml_mod

    text = yaml_mod.safe_dump(report.to_mapping(), sort_keys=False)
    assert "SECRET_VALUE_abc123" not in text
    assert "ANOTHER_SECRET" not in text
    assert "a leading comment" not in text
    assert "trailing comment" not in text


def test_yaml_parse_failure_raises_structure_extract_error() -> None:
    bad = "policies:\n  unterminated: [abc\n"
    with pytest.raises(StructureExtractError) as excinfo:
        extract_structure(bad, **_args())
    assert "YAML parse error" in str(excinfo.value)


def test_yaml_empty_source_is_empty_report() -> None:
    report = extract_structure("", **_args())
    assert report.items == []


def test_report_serialization_is_deterministic() -> None:
    import yaml as yaml_mod

    r1 = extract_structure(_YAML_SOURCE, **_args())
    r2 = extract_structure(_YAML_SOURCE, **_args())
    text1 = yaml_mod.safe_dump(r1.to_mapping(), sort_keys=False)
    text2 = yaml_mod.safe_dump(r2.to_mapping(), sort_keys=False)
    assert text1 == text2
    # And repeated calls produce byte-identical items in the same
    # order (stable symbol_path / locator values).
    assert [i.to_mapping() for i in r1.items] == [i.to_mapping() for i in r2.items]


def test_report_has_no_timestamp_or_op_id() -> None:
    import yaml as yaml_mod

    report = extract_structure(_YAML_SOURCE, **_args())
    text = yaml_mod.safe_dump(report.to_mapping(), sort_keys=False)
    for forbidden in ("timestamp", "op_id", "generated_at", "created_at"):
        assert forbidden not in text, (
            f"report leaks environment-dependent field {forbidden!r}: {text}"
        )


def test_write_structure_report_atomic_and_roundtrips(tmp_path) -> None:
    from llloom.workspace.layout import Workspace
    import yaml as yaml_mod

    ws = Workspace.init(tmp_path)
    report = extract_structure(_YAML_SOURCE, **_args())
    out_path = write_structure_report(ws, report)
    assert out_path == ws.structure_report_path("src.policies")
    assert out_path.is_file()
    # No stale .tmp leftover.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    assert not tmp.exists()
    # Round-trip through YAML preserves the report mapping.
    loaded = yaml_mod.safe_load(out_path.read_text(encoding="utf-8"))
    assert loaded["version"] == "structure_report_v1"
    assert loaded["source_id"] == "src.policies"
    assert loaded["language"] == "yaml"


# --- unsupported classes and wrong locator_type ------------------------


def test_unsupported_source_class_raises() -> None:
    with pytest.raises(StructureExtractError):
        extract_structure(_YAML_SOURCE, **_args(source_class="markdown_prose"))


def test_wrong_locator_type_raises() -> None:
    with pytest.raises(StructureExtractError):
        extract_structure(_YAML_SOURCE, **_args(locator_type="markdown_prose_v1"))


# --- Python extraction: lazy-import refusal when tree-sitter missing ----


def test_python_extraction_requires_structured_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a machine without tree-sitter installed. The adapter
    must raise ``StructureExtractError`` whose message names
    ``llloom[structured]``."""

    class _Blocker:
        def find_spec(self, name, path=None, target=None):  # noqa: ARG002
            if name == "tree_sitter" or name.startswith("tree_sitter"):
                raise ImportError(f"no {name}")
            return None

    for mod_name in list(sys.modules):
        if mod_name == "tree_sitter" or mod_name.startswith("tree_sitter"):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])

    with pytest.raises(StructureExtractError) as excinfo:
        extract_structure(
            "def foo():\n    return 1\n",
            **_args(
                source_class="code",
                raw_path="raw/sources/foo.py",
            ),
        )
    assert "llloom[structured]" in str(excinfo.value)


# --- Broader-language `code` slice: suffix dispatch + fakes -------------


class _FakeNode:
    """Minimal tree-sitter node mock for offline unit tests.

    Mimics the subset of the tree-sitter ``Node`` interface that the
    walker reads: ``type``, ``start_byte``/``end_byte``,
    ``start_point``/``end_point``, ``children``, and
    ``child_by_field_name``.
    """

    def __init__(
        self,
        ts_type: str,
        *,
        start_point: tuple[int, int] = (0, 0),
        end_point: tuple[int, int] = (0, 1),
        start_byte: int = 0,
        end_byte: int = 0,
        children=None,
        fields=None,
    ) -> None:
        self.type = ts_type
        self.start_point = start_point
        self.end_point = end_point
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.children = list(children or [])
        self._fields = dict(fields or {})

    def child_by_field_name(self, name: str):
        return self._fields.get(name)


class _FakeTree:
    def __init__(self, root_node: _FakeNode) -> None:
        self.root_node = root_node


class _FakeParser:
    """Parser stub that returns a pre-built ``_FakeTree``."""

    def __init__(self, root: _FakeNode) -> None:
        self._root = root

    def parse(self, _data: bytes):
        return _FakeTree(self._root)


def _build_named_node(
    ts_type: str,
    name: str,
    *,
    source: str,
    children=None,
    fields=None,
    identifier_ts_type: str = "identifier",
) -> _FakeNode:
    """Build a fake node carrying a ``name`` field whose byte range
    points at ``name`` inside ``source``.

    Tests rely on ``_named_field_or_identifier`` reading byte offsets
    into the original source string passed to ``extract_structure``,
    so the byte range must reference a real occurrence of ``name`` in
    that string.
    """
    name_offset = source.index(name)
    name_node = _FakeNode(
        identifier_ts_type,
        start_byte=name_offset,
        end_byte=name_offset + len(name),
    )
    return _FakeNode(
        ts_type,
        children=list(children or []) + [name_node],
        fields={"name": name_node, **(fields or {})},
    )


def test_unsupported_code_suffix_refuses_with_supported_message() -> None:
    """`.tsx`, `.js`, `.kt`, etc. are explicitly out of scope; the
    error names the supported suffixes and the install extra.

    Slice 082 added ``.java`` to the supported set, so this test now
    pins ``.java`` as supported alongside the others and uses
    ``.tsx`` as the unsupported example.
    """
    with pytest.raises(StructureExtractError) as excinfo:
        extract_structure(
            "function foo(): void {}\n",
            **_args(source_class="code", raw_path="raw/sources/foo.tsx"),
        )
    msg = str(excinfo.value)
    assert "'.tsx'" in msg
    for suffix in (".py", ".go", ".rs", ".ts", ".cs", ".java"):
        assert suffix in msg
    assert "llloom[structured]" in msg


def _block_tree_sitter(
    monkeypatch: pytest.MonkeyPatch, *, language_modules: set[str]
) -> None:
    """Block import of ``tree_sitter_<lang>`` modules while leaving
    ``tree_sitter`` itself importable. This proves the failure occurs
    at the language-specific import, not at the core import.
    """
    blocked = {f"tree_sitter_{name}" for name in language_modules}

    class _Blocker:
        def find_spec(self, name, path=None, target=None):  # noqa: ARG002
            if name in blocked:
                raise ImportError(f"no {name}")
            return None

    for mod_name in list(sys.modules):
        if mod_name in blocked:
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])


def test_go_extraction_requires_structured_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_tree_sitter(monkeypatch, language_modules={"go"})
    with pytest.raises(StructureExtractError) as excinfo:
        extract_structure(
            "package main\nfunc Main() {}\n",
            **_args(source_class="code", raw_path="raw/sources/main.go"),
        )
    assert "llloom[structured]" in str(excinfo.value)


def test_rust_extraction_requires_structured_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_tree_sitter(monkeypatch, language_modules={"rust"})
    with pytest.raises(StructureExtractError) as excinfo:
        extract_structure(
            "fn main() {}\n",
            **_args(source_class="code", raw_path="raw/sources/main.rs"),
        )
    assert "llloom[structured]" in str(excinfo.value)


def test_typescript_extraction_requires_structured_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_tree_sitter(monkeypatch, language_modules={"typescript"})
    with pytest.raises(StructureExtractError) as excinfo:
        extract_structure(
            "export function foo(): void {}\n",
            **_args(source_class="code", raw_path="raw/sources/foo.ts"),
        )
    assert "llloom[structured]" in str(excinfo.value)


def test_go_positive_extraction_with_fake_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build a tiny fake Go syntax tree and assert the walker emits
    the expected ``(kind, symbol_path)`` set for top-level functions,
    methods, and named types."""
    source = (
        "package main\n"
        "func BuildIndex() {}\n"
        "type Config struct{}\n"
        "func (s *Store) Save() {}\n"
    )

    func_decl = _build_named_node("function_declaration", "BuildIndex", source=source)
    type_spec = _build_named_node(
        "type_spec", "Config", source=source, identifier_ts_type="type_identifier"
    )
    type_decl = _FakeNode("type_declaration", children=[type_spec])

    # method_declaration with a receiver field pointing at "Store".
    receiver_type_offset = source.index("Store")
    receiver_type_node = _FakeNode(
        "type_identifier",
        start_byte=receiver_type_offset,
        end_byte=receiver_type_offset + len("Store"),
    )
    receiver_list = _FakeNode(
        "parameter_list",
        children=[
            _FakeNode(
                "parameter_declaration",
                children=[
                    _FakeNode("pointer_type", children=[receiver_type_node]),
                ],
            )
        ],
    )
    method_decl = _build_named_node(
        "method_declaration",
        "Save",
        source=source,
        children=[receiver_list],
        fields={"receiver": receiver_list},
    )

    root = _FakeNode(
        "source_file",
        children=[func_decl, type_decl, method_decl],
    )
    fake_parser = _FakeParser(root)

    import llloom.structured.extract as ext

    monkeypatch.setattr(ext, "_load_go_parser", lambda: (fake_parser, object()))

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/main.go"),
    )
    assert report.language == "go"
    by_path = {(it.symbol_path, it.kind) for it in report.items}
    assert by_path == {
        ("BuildIndex", "function"),
        ("Config", "type"),
        ("Store.Save", "method"),
    }


def test_rust_and_typescript_positive_extraction_with_fake_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust: top-level fn + struct + enum + trait + impl-method.
    TypeScript: function + class + interface + type alias + class
    method. Both languages prove the broader-language walker emits
    the expected ``(kind, symbol_path)`` sets."""
    import llloom.structured.extract as ext

    # ---- Rust ---------------------------------------------------------
    rust_source = (
        "fn build_index() {}\n"
        "struct Config;\n"
        "enum Mode { On, Off }\n"
        "trait Store {}\n"
        "impl Store { fn save(&self) {} }\n"
    )
    fn_item = _build_named_node("function_item", "build_index", source=rust_source)
    struct_item = _build_named_node(
        "struct_item",
        "Config",
        source=rust_source,
        identifier_ts_type="type_identifier",
    )
    enum_item = _build_named_node(
        "enum_item",
        "Mode",
        source=rust_source,
        identifier_ts_type="type_identifier",
    )
    trait_item = _build_named_node(
        "trait_item",
        "Store",
        source=rust_source,
        identifier_ts_type="type_identifier",
    )
    # impl_item: ``type`` field names the target; one function_item
    # nested as a child.
    target_offset = rust_source.rindex("Store")  # the impl-target Store
    impl_target = _FakeNode(
        "type_identifier",
        start_byte=target_offset,
        end_byte=target_offset + len("Store"),
    )
    save_fn = _build_named_node("function_item", "save", source=rust_source)
    impl_item = _FakeNode(
        "impl_item",
        children=[impl_target, save_fn],
        fields={"type": impl_target},
    )
    rust_root = _FakeNode(
        "source_file",
        children=[fn_item, struct_item, enum_item, trait_item, impl_item],
    )
    monkeypatch.setattr(
        ext, "_load_rust_parser", lambda: (_FakeParser(rust_root), object())
    )
    rust_report = extract_structure(
        rust_source,
        **_args(source_class="code", raw_path="raw/sources/lib.rs"),
    )
    assert rust_report.language == "rust"
    rust_set = {(it.symbol_path, it.kind) for it in rust_report.items}
    assert rust_set == {
        ("build_index", "function"),
        ("Config", "struct"),
        ("Mode", "enum"),
        ("Store", "trait"),
        ("Store.save", "method"),
    }

    # ---- TypeScript ---------------------------------------------------
    ts_source = (
        "export function buildIndex(): void {}\n"
        "export class Store { save() {} }\n"
        "export interface BuildOptions {}\n"
        "export type Mode = 'on' | 'off';\n"
    )
    fn_decl = _build_named_node(
        "function_declaration", "buildIndex", source=ts_source
    )
    method_def = _build_named_node(
        "method_definition",
        "save",
        source=ts_source,
        identifier_ts_type="property_identifier",
    )
    class_decl = _build_named_node(
        "class_declaration",
        "Store",
        source=ts_source,
        children=[method_def],
        identifier_ts_type="type_identifier",
    )
    interface_decl = _build_named_node(
        "interface_declaration",
        "BuildOptions",
        source=ts_source,
        identifier_ts_type="type_identifier",
    )
    type_alias = _build_named_node(
        "type_alias_declaration",
        "Mode",
        source=ts_source,
        identifier_ts_type="type_identifier",
    )
    ts_root = _FakeNode(
        "program",
        children=[fn_decl, class_decl, interface_decl, type_alias],
    )
    monkeypatch.setattr(
        ext, "_load_typescript_parser", lambda: (_FakeParser(ts_root), object())
    )
    ts_report = extract_structure(
        ts_source,
        **_args(source_class="code", raw_path="raw/sources/index.ts"),
    )
    assert ts_report.language == "typescript"
    ts_set = {(it.symbol_path, it.kind) for it in ts_report.items}
    assert ts_set == {
        ("buildIndex", "function"),
        ("Store", "class"),
        ("Store.save", "method"),
        ("BuildOptions", "interface"),
        ("Mode", "type_alias"),
    }


def test_csharp_extraction_requires_structured_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_tree_sitter(monkeypatch, language_modules={"c_sharp"})
    with pytest.raises(StructureExtractError) as excinfo:
        extract_structure(
            "class Store {}\n",
            **_args(source_class="code", raw_path="raw/sources/Store.cs"),
        )
    assert "llloom[structured]" in str(excinfo.value)


def test_csharp_positive_extraction_with_fake_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build a tiny fake C# syntax tree and assert the walker emits
    the expected ``(symbol_path, kind)`` set for the narrow first
    declaration set (class / interface / struct / enum / method).
    Methods inside a class inherit the container's prefix via the
    generic walker, so an inner ``Save`` shows up as ``Store.Save``."""
    source = (
        "class Store {\n"
        "    void Save() {}\n"
        "}\n"
        "interface IService {}\n"
        "struct Vec2 {}\n"
        "enum Mode { On, Off }\n"
    )
    method_decl = _build_named_node(
        "method_declaration", "Save", source=source
    )
    class_decl = _build_named_node(
        "class_declaration",
        "Store",
        source=source,
        children=[method_decl],
    )
    interface_decl = _build_named_node(
        "interface_declaration", "IService", source=source
    )
    struct_decl = _build_named_node(
        "struct_declaration", "Vec2", source=source
    )
    enum_decl = _build_named_node(
        "enum_declaration", "Mode", source=source
    )
    root = _FakeNode(
        "compilation_unit",
        children=[class_decl, interface_decl, struct_decl, enum_decl],
    )
    fake_parser = _FakeParser(root)

    import llloom.structured.extract as ext

    monkeypatch.setattr(
        ext, "_load_csharp_parser", lambda: (fake_parser, object())
    )

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/Store.cs"),
    )
    assert report.language == "csharp"
    cs_set = {(it.symbol_path, it.kind) for it in report.items}
    assert cs_set == {
        ("Store", "class"),
        ("Store.Save", "method"),
        ("IService", "interface"),
        ("Vec2", "struct"),
        ("Mode", "enum"),
    }


# ---- Unity bridge v1: direct MonoBehaviour subclass detection ----------


def _csharp_class_with_base(
    class_name: str,
    base_text: str,
    *,
    source: str,
    base_entry_type: str = "identifier",
    extra_children=None,
) -> _FakeNode:
    """Build a fake C# `class_declaration` node carrying both a
    ``name`` field and a ``bases`` field whose `base_list` contains
    exactly one direct base entry with byte range pointing at
    ``base_text`` inside ``source``."""
    name_offset = source.index(class_name)
    name_node = _FakeNode(
        "identifier",
        start_byte=name_offset,
        end_byte=name_offset + len(class_name),
    )
    base_offset = source.index(base_text)
    base_entry = _FakeNode(
        base_entry_type,
        start_byte=base_offset,
        end_byte=base_offset + len(base_text),
    )
    base_list = _FakeNode("base_list", children=[base_entry])
    children = [name_node, base_list] + list(extra_children or [])
    return _FakeNode(
        "class_declaration",
        children=children,
        fields={"name": name_node, "bases": base_list},
    )


def test_csharp_direct_monobehaviour_base_yields_unity_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct `class PlayerController : MonoBehaviour` declaration
    surfaces as ``kind == "unity_component"`` under the Unity bridge."""
    source = "class PlayerController : MonoBehaviour {}\n"
    class_node = _csharp_class_with_base(
        "PlayerController", "MonoBehaviour", source=source
    )
    root = _FakeNode("compilation_unit", children=[class_node])

    import llloom.structured.extract as ext

    monkeypatch.setattr(
        ext, "_load_csharp_parser", lambda: (_FakeParser(root), object())
    )

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/PlayerController.cs"),
    )
    assert report.language == "csharp"
    assert {(it.symbol_path, it.kind) for it in report.items} == {
        ("PlayerController", "unity_component"),
    }


def test_csharp_qualified_unityengine_monobehaviour_base_yields_unity_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A qualified base `UnityEngine.MonoBehaviour` also yields
    ``unity_component``: the textual rule admits any base entry
    whose stripped text equals `MonoBehaviour` or ends with
    `.MonoBehaviour`."""
    source = "class CameraRig : UnityEngine.MonoBehaviour {}\n"
    class_node = _csharp_class_with_base(
        "CameraRig",
        "UnityEngine.MonoBehaviour",
        source=source,
        base_entry_type="qualified_name",
    )
    root = _FakeNode("compilation_unit", children=[class_node])

    import llloom.structured.extract as ext

    monkeypatch.setattr(
        ext, "_load_csharp_parser", lambda: (_FakeParser(root), object())
    )

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/CameraRig.cs"),
    )
    assert {(it.symbol_path, it.kind) for it in report.items} == {
        ("CameraRig", "unity_component"),
    }


def test_csharp_normal_class_and_unrelated_base_remain_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A class with no base list still yields `class`; a class
    inheriting from something other than `MonoBehaviour` (e.g.
    `ScriptableObject`, an unrelated `Object`, an indirect chain via
    another type) also yields `class`. Unity bridge v1 is strictly
    direct-base only — `ScriptableObject` and any non-MonoBehaviour
    base remain deferred."""
    source = (
        "class Bare {}\n"
        "class CustomAsset : ScriptableObject {}\n"
        "class Indirect : MyBase {}\n"
    )
    bare_node = _build_named_node("class_declaration", "Bare", source=source)
    scriptable_node = _csharp_class_with_base(
        "CustomAsset", "ScriptableObject", source=source
    )
    indirect_node = _csharp_class_with_base(
        "Indirect", "MyBase", source=source
    )
    root = _FakeNode(
        "compilation_unit",
        children=[bare_node, scriptable_node, indirect_node],
    )

    import llloom.structured.extract as ext

    monkeypatch.setattr(
        ext, "_load_csharp_parser", lambda: (_FakeParser(root), object())
    )

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/Mixed.cs"),
    )
    cs_set = {(it.symbol_path, it.kind) for it in report.items}
    assert cs_set == {
        ("Bare", "class"),
        ("CustomAsset", "class"),
        ("Indirect", "class"),
    }


def test_csharp_unity_component_supports_nested_method_symbol_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `unity_component` class still participates in normal nested
    symbol-path behavior: a `method_declaration` inside it inherits
    the container prefix and surfaces as
    ``PlayerController.HandleJump`` with `kind == "method"`. Methods
    are not re-tagged as Unity-flavored; only the class itself is."""
    source = (
        "class PlayerController : MonoBehaviour {\n"
        "    void HandleJump() {}\n"
        "}\n"
    )
    method_decl = _build_named_node(
        "method_declaration", "HandleJump", source=source
    )
    class_node = _csharp_class_with_base(
        "PlayerController",
        "MonoBehaviour",
        source=source,
        extra_children=[method_decl],
    )
    root = _FakeNode("compilation_unit", children=[class_node])

    import llloom.structured.extract as ext

    monkeypatch.setattr(
        ext, "_load_csharp_parser", lambda: (_FakeParser(root), object())
    )

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/PlayerController.cs"),
    )
    cs_set = {(it.symbol_path, it.kind) for it in report.items}
    assert cs_set == {
        ("PlayerController", "unity_component"),
        ("PlayerController.HandleJump", "method"),
    }


# ---- Java (Slice 082) -------------------------------------------------


def test_java_extraction_requires_structured_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block ``tree_sitter_java`` while keeping a minimal
    ``tree_sitter`` stub importable, then assert the loader's
    missing-extra refusal names ``tree-sitter-java``, ``.java``, and
    ``llloom[structured]``.

    The stub is necessary because the dev venv runs without the
    optional ``llloom[structured]`` extra; without it, the first
    ``from tree_sitter import Parser`` would fail before the
    Java-specific path runs and the test could only verify the
    generic core-missing message. Stubbing ``tree_sitter`` makes the
    second loader path fire deterministically and pins the
    Java-specific diagnostic shape the dev note asked for.

    The package-level import of ``llloom.structured.extract``
    already proved Java grammar imports are lazy (every
    Java-specific import lives inside ``_load_java_parser``).
    """
    # Minimal ``tree_sitter`` stub so the loader's first try-block
    # succeeds; the second try-block must then raise the Java-
    # specific StructureExtractError.
    tree_sitter_stub = types.ModuleType("tree_sitter")

    class _StubParser:
        def parse(self, _data):
            raise AssertionError(
                "stub parser must never be called; the Java loader "
                "should raise before reaching parse()"
            )

    tree_sitter_stub.Parser = _StubParser  # type: ignore[attr-defined]
    tree_sitter_stub.Language = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tree_sitter", tree_sitter_stub)

    _block_tree_sitter(monkeypatch, language_modules={"java"})
    with pytest.raises(StructureExtractError) as excinfo:
        extract_structure(
            "class Project {}\n",
            **_args(source_class="code", raw_path="raw/sources/Project.java"),
        )
    msg = str(excinfo.value)
    assert "tree-sitter-java" in msg
    assert ".java" in msg
    assert "llloom[structured]" in msg


def test_java_positive_extraction_with_fake_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build a tiny fake Java syntax tree carrying every V1 item kind
    (class, interface, enum, record, method, constructor, field) and
    assert the walker emits the expected ``(symbol_path, kind)`` set.

    The source also plants distinctive bodies / comments / scalar
    literals to verify metadata-only behavior: none of them must
    appear in the serialized report.
    """
    source = (
        "package com.example;\n"
        "\n"
        "import java.util.List;\n"
        "\n"
        "public class Project {\n"
        "    private int maxsize = SECRET_LITERAL_99;\n"
        "    public Project() { System.out.println(\"NOISE_CTOR_42\"); }\n"
        "    public int evaluate() { return COMMENT_PLANT; }\n"
        "}\n"
        "\n"
        "interface Metric {}\n"
        "\n"
        "enum PatchKind { LAND, WATER }\n"
        "\n"
        "record PatchRecord(int id) {}\n"
    )

    # Field declaration: tree-sitter-java wraps the declarator (the
    # field-name carrier) under the ``field_declaration`` node.
    maxsize_offset = source.index("maxsize")
    maxsize_id = _FakeNode(
        "identifier",
        start_byte=maxsize_offset,
        end_byte=maxsize_offset + len("maxsize"),
    )
    maxsize_declarator = _FakeNode(
        "variable_declarator",
        children=[maxsize_id],
        fields={"name": maxsize_id},
    )
    field_decl = _FakeNode(
        "field_declaration",
        start_point=(5, 4),
        end_point=(5, 40),
        children=[maxsize_declarator],
    )

    # Constructor declaration: its ``name`` field carries an
    # identifier whose text equals the class name.
    ctor_name_offset = source.index("public Project()") + len("public ")
    ctor_name = _FakeNode(
        "identifier",
        start_byte=ctor_name_offset,
        end_byte=ctor_name_offset + len("Project"),
    )
    ctor_decl = _FakeNode(
        "constructor_declaration",
        start_point=(6, 4),
        end_point=(6, 64),
        children=[ctor_name],
        fields={"name": ctor_name},
    )

    evaluate_decl = _build_named_node(
        "method_declaration", "evaluate", source=source
    )
    # Re-anchor evaluate's body to a higher line so the deterministic
    # sort by (start_line, start_col, symbol_path) keeps it after the
    # class header but before sibling top-level declarations.
    evaluate_decl.start_point = (7, 4)
    evaluate_decl.end_point = (7, 50)

    class_decl = _build_named_node(
        "class_declaration",
        "Project",
        source=source,
        children=[field_decl, ctor_decl, evaluate_decl],
    )
    class_decl.start_point = (4, 0)
    class_decl.end_point = (8, 1)

    interface_decl = _build_named_node(
        "interface_declaration", "Metric", source=source
    )
    interface_decl.start_point = (10, 0)
    interface_decl.end_point = (10, 18)

    enum_decl = _build_named_node(
        "enum_declaration", "PatchKind", source=source
    )
    enum_decl.start_point = (12, 0)
    enum_decl.end_point = (12, 30)

    record_decl = _build_named_node(
        "record_declaration", "PatchRecord", source=source
    )
    record_decl.start_point = (14, 0)
    record_decl.end_point = (14, 30)

    root = _FakeNode(
        "program",
        children=[class_decl, interface_decl, enum_decl, record_decl],
    )

    import llloom.structured.extract as ext

    monkeypatch.setattr(
        ext, "_load_java_parser", lambda: (_FakeParser(root), object())
    )

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/Project.java"),
    )
    assert report.language == "java"
    by_path = {(it.symbol_path, it.kind) for it in report.items}
    assert by_path == {
        ("Project", "class"),
        ("Project.Project", "constructor"),
        ("Project.evaluate", "method"),
        ("Project.maxsize", "field"),
        ("Metric", "interface"),
        ("PatchKind", "enum"),
        ("PatchRecord", "record"),
    }

    # Metadata-only invariant: none of the planted body, comment, or
    # literal strings appears in the serialized report.
    import yaml as yaml_mod

    text = yaml_mod.safe_dump(report.to_mapping(), sort_keys=False)
    for forbidden in (
        "SECRET_LITERAL_99",
        "NOISE_CTOR_42",
        "COMMENT_PLANT",
        "System.out.println",
        "import java.util.List",
    ):
        assert forbidden not in text, (
            f"metadata-only walker leaked {forbidden!r}: {text}"
        )
