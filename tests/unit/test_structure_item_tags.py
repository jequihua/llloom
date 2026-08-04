"""Unit tests for the generic framework-tagging metadata channel
on :class:`StructureItem` (planned package milestone landed
2026-05-14).

Pins the new contract:

- `StructureItem.tags` defaults to `()` so every existing
  extractor path produces empty-tagged items;
- ordinary C# `class`, `interface`, `struct`, `enum`, and `method`
  declarations remain empty-tagged;
- a direct C# `MonoBehaviour` subclass keeps `kind ==
  "unity_component"` AND additionally carries the generic tag pair
  `("framework:unity", "role:component")`;
- a qualified `UnityEngine.MonoBehaviour` subclass behaves the
  same way (the existing textual base-list rule is unchanged);
- methods inside a Unity component do NOT inherit the class
  tags — only the class itself carries framework metadata;
- serialized structure reports (`to_mapping` → YAML) include the
  tags as a list (empty list for ordinary items);
- legacy serialized items without a `tags` key rehydrate to `()`
  through `StructureItem.from_mapping`, preserving backward
  compatibility with `structure_report_v1` reports written before
  this slice.

The C# tests reuse the fake-tree-sitter machinery already in
`test_structure_extract.py` so the suite still passes without the
optional `llloom[structured]` extra installed.
"""

from __future__ import annotations

import yaml

from llloom.structured.extract import (
    StructureItem,
    extract_structure,
)
from tests.unit.test_structure_extract import (  # type: ignore[import-not-found]
    _FakeNode,
    _FakeParser,
    _args,
    _build_named_node,
    _csharp_class_with_base,
)


# ---- defaults ------------------------------------------------------------


def test_structure_item_default_tags_is_empty_tuple() -> None:
    item = StructureItem(
        kind="class",
        name="Foo",
        symbol_path="Foo",
        locator={"locator_type": "code_v1"},
    )
    assert item.tags == ()


def test_structure_item_to_mapping_includes_empty_tags_list() -> None:
    item = StructureItem(
        kind="class",
        name="Foo",
        symbol_path="Foo",
        locator={"locator_type": "code_v1"},
    )
    mapping = item.to_mapping()
    assert mapping["tags"] == []


def test_structure_item_to_mapping_includes_non_empty_tags() -> None:
    item = StructureItem(
        kind="unity_component",
        name="Player",
        symbol_path="Player",
        locator={"locator_type": "code_v1"},
        tags=("framework:unity", "role:component"),
    )
    mapping = item.to_mapping()
    assert mapping["tags"] == ["framework:unity", "role:component"]


# ---- legacy / minimal mapping rehydration --------------------------------


def test_from_mapping_normalizes_missing_tags_to_empty_tuple() -> None:
    """`structure_report_v1` reports written before this slice did
    not carry a `tags` key; `from_mapping` must accept those
    mappings and produce an item with `tags == ()`."""
    legacy = {
        "kind": "class",
        "name": "OldClass",
        "symbol_path": "OldClass",
        "locator": {"locator_type": "code_v1", "path": "x.cs"},
    }
    item = StructureItem.from_mapping(legacy)
    assert item.tags == ()
    assert item.kind == "class"
    assert item.name == "OldClass"


def test_from_mapping_normalizes_none_tags_to_empty_tuple() -> None:
    legacy = {
        "kind": "class",
        "name": "OldClass",
        "symbol_path": "OldClass",
        "locator": {"locator_type": "code_v1"},
        "tags": None,
    }
    item = StructureItem.from_mapping(legacy)
    assert item.tags == ()


def test_from_mapping_preserves_string_tags_and_drops_non_strings() -> None:
    data = {
        "kind": "unity_component",
        "name": "Player",
        "symbol_path": "Player",
        "locator": {"locator_type": "code_v1"},
        "tags": ["framework:unity", "role:component", 7, None, "extra:value"],
    }
    item = StructureItem.from_mapping(data)
    assert item.tags == ("framework:unity", "role:component", "extra:value")


def test_to_mapping_roundtrips_through_from_mapping() -> None:
    original = StructureItem(
        kind="unity_component",
        name="Player",
        symbol_path="Player",
        locator={"locator_type": "code_v1", "path": "Player.cs"},
        tags=("framework:unity", "role:component"),
    )
    rehydrated = StructureItem.from_mapping(original.to_mapping())
    assert rehydrated == original


# ---- C# extractor: ordinary declarations stay empty-tagged --------------


def _setup_csharp_extractor(monkeypatch, root: _FakeNode) -> None:
    import llloom.structured.extract as ext

    monkeypatch.setattr(
        ext, "_load_csharp_parser", lambda: (_FakeParser(root), object())
    )


def test_csharp_ordinary_class_has_empty_tags(monkeypatch) -> None:
    source = "class Store {}\n"
    class_node = _build_named_node("class_declaration", "Store", source=source)
    root = _FakeNode("compilation_unit", children=[class_node])
    _setup_csharp_extractor(monkeypatch, root)

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/Store.cs"),
    )
    items_by_path = {it.symbol_path: it for it in report.items}
    assert items_by_path["Store"].kind == "class"
    assert items_by_path["Store"].tags == ()


def test_csharp_ordinary_interface_struct_enum_method_have_empty_tags(
    monkeypatch,
) -> None:
    source = (
        "interface IStore {}\n"
        "struct Point {}\n"
        "enum Color {}\n"
    )
    interface_node = _build_named_node(
        "interface_declaration", "IStore", source=source
    )
    struct_node = _build_named_node(
        "struct_declaration", "Point", source=source
    )
    enum_node = _build_named_node("enum_declaration", "Color", source=source)
    root = _FakeNode(
        "compilation_unit",
        children=[interface_node, struct_node, enum_node],
    )
    _setup_csharp_extractor(monkeypatch, root)

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/Mixed.cs"),
    )
    by_path = {it.symbol_path: it for it in report.items}
    assert by_path["IStore"].tags == ()
    assert by_path["Point"].tags == ()
    assert by_path["Color"].tags == ()


# ---- C# extractor: Unity bridge tags ------------------------------------


def test_csharp_direct_monobehaviour_subclass_carries_framework_tags(
    monkeypatch,
) -> None:
    """A direct `class PlayerController : MonoBehaviour` keeps
    `kind == "unity_component"` (backward compatibility) and
    additionally carries the generic framework tag pair."""
    source = "class PlayerController : MonoBehaviour {}\n"
    class_node = _csharp_class_with_base(
        "PlayerController", "MonoBehaviour", source=source
    )
    root = _FakeNode("compilation_unit", children=[class_node])
    _setup_csharp_extractor(monkeypatch, root)

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/PlayerController.cs"),
    )
    by_path = {it.symbol_path: it for it in report.items}
    item = by_path["PlayerController"]
    assert item.kind == "unity_component"
    assert item.tags == ("framework:unity", "role:component")


def test_csharp_qualified_unityengine_monobehaviour_carries_framework_tags(
    monkeypatch,
) -> None:
    source = "class CameraRig : UnityEngine.MonoBehaviour {}\n"
    class_node = _csharp_class_with_base(
        "CameraRig",
        "UnityEngine.MonoBehaviour",
        source=source,
        base_entry_type="qualified_name",
    )
    root = _FakeNode("compilation_unit", children=[class_node])
    _setup_csharp_extractor(monkeypatch, root)

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/CameraRig.cs"),
    )
    by_path = {it.symbol_path: it for it in report.items}
    item = by_path["CameraRig"]
    assert item.kind == "unity_component"
    assert item.tags == ("framework:unity", "role:component")


def test_csharp_method_inside_unity_component_does_not_inherit_tags(
    monkeypatch,
) -> None:
    """A `method_declaration` nested inside a Unity-component class
    must NOT carry the framework / role tags. Only the class
    declaration itself carries the framework metadata; methods stay
    empty-tagged."""
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
    _setup_csharp_extractor(monkeypatch, root)

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/PlayerController.cs"),
    )
    by_path = {it.symbol_path: it for it in report.items}
    assert by_path["PlayerController"].kind == "unity_component"
    assert by_path["PlayerController"].tags == ("framework:unity", "role:component")
    # The method emitted inside a Unity component must stay tag-free.
    assert by_path["PlayerController.HandleJump"].kind == "method"
    assert by_path["PlayerController.HandleJump"].tags == ()


def test_csharp_class_inheriting_unrelated_base_has_empty_tags(
    monkeypatch,
) -> None:
    """A class that inherits from something other than
    `MonoBehaviour` must remain `kind == "class"` and empty-tagged
    — the classifier is intentionally narrow."""
    source = "class Store : BaseStore {}\n"
    class_node = _csharp_class_with_base(
        "Store", "BaseStore", source=source
    )
    root = _FakeNode("compilation_unit", children=[class_node])
    _setup_csharp_extractor(monkeypatch, root)

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/Store.cs"),
    )
    by_path = {it.symbol_path: it for it in report.items}
    assert by_path["Store"].kind == "class"
    assert by_path["Store"].tags == ()


# ---- serialized report includes tags ------------------------------------


def test_serialized_structure_report_includes_tags_field_on_every_item(
    monkeypatch,
) -> None:
    """When the extractor emits a Unity-component class alongside an
    ordinary class, the serialized YAML structure report carries a
    `tags` list on every item: populated on the Unity component,
    empty on the ordinary class."""
    source = (
        "class PlayerController : MonoBehaviour {}\n"
        "class Store {}\n"
    )
    unity_node = _csharp_class_with_base(
        "PlayerController", "MonoBehaviour", source=source
    )
    plain_node = _build_named_node(
        "class_declaration", "Store", source=source
    )
    root = _FakeNode("compilation_unit", children=[unity_node, plain_node])
    _setup_csharp_extractor(monkeypatch, root)

    report = extract_structure(
        source,
        **_args(source_class="code", raw_path="raw/sources/Mixed.cs"),
    )
    serialized = yaml.safe_dump(report.to_mapping(), sort_keys=False)
    parsed = yaml.safe_load(serialized)

    assert parsed["version"] == "structure_report_v1"
    items = {it["symbol_path"]: it for it in parsed["items"]}
    assert items["PlayerController"]["tags"] == [
        "framework:unity",
        "role:component",
    ]
    # Every item carries the tags key, even when empty — the field is
    # not optional in the serialized form.
    for it in parsed["items"]:
        assert "tags" in it
        assert isinstance(it["tags"], list)
    assert items["Store"]["tags"] == []
