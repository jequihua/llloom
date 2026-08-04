"""Deterministic structured-source extractor.

Given raw source text, emits a compact :class:`StructureReport` with
structure-only items (key paths, symbols, kinds, ``code_v1``
locators) and an atomic writer that persists the report under
``state/structure/<source_id>.yaml``. The extractor never calls
``LLMInvoke`` and never stores scalar values, comments, docstrings,
full source lines, or code bodies.

The YAML path uses PyYAML's ``compose`` API, which is already in the
base install. The Python path lazy-imports tree-sitter inside
``_extract_python_structure`` so the module is safe to import without
the ``llloom[structured]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llloom.workspace.layout import Workspace


REPORT_VERSION = "structure_report_v1"


SUPPORTED_SOURCE_CLASSES: frozenset[str] = frozenset(
    {"structured_yaml", "code"}
)


class StructureExtractError(Exception):
    """Raised when a structured-source extraction cannot be completed.

    Typical causes: unsupported source class, malformed source text,
    or a missing optional dependency required by the chosen
    extractor (for example, the ``llloom[structured]`` extra is
    required for Python/code extraction).
    """


@dataclass(frozen=True)
class StructureItem:
    """One item in a structure report.

    ``symbol_path`` is a dot-delimited path inside the source (e.g.
    ``"policies.markdown_prose"`` for a nested YAML key, or
    ``"PolicyLoader.resolve_policy"`` for a Python class method).
    ``locator`` is a ``code_v1`` mapping the verifier can resolve.

    ``tags`` is a generic, additive metadata channel for framework /
    role classification produced by deterministic classifiers in the
    extractor. Tags are lowercase ASCII ``prefix:value`` strings and
    default to ``()`` (empty). They never replace ``kind``, never
    drive verifier, render, or claim semantics, and are not
    user-configurable in this slice. The first shipped classifier
    attaches ``framework:unity`` and ``role:component`` to direct C#
    ``MonoBehaviour`` subclasses (whose ``kind`` remains
    ``unity_component`` for backward compatibility with existing
    consumers).
    """

    kind: str
    name: str
    symbol_path: str
    locator: dict[str, Any]
    tags: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "symbol_path": self.symbol_path,
            "locator": dict(self.locator),
            "tags": list(self.tags),
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "StructureItem":
        """Rehydrate a :class:`StructureItem` from a serialized mapping.

        Absent or ``None`` ``tags`` normalizes to ``()`` so older
        ``structure_report_v1`` reports written before this slice
        still round-trip cleanly. Non-string tag entries are dropped
        defensively rather than raising.
        """
        raw_tags = data.get("tags") or []
        if not isinstance(raw_tags, (list, tuple)):
            raw_tags = []
        tags = tuple(t for t in raw_tags if isinstance(t, str))
        return cls(
            kind=str(data["kind"]),
            name=str(data["name"]),
            symbol_path=str(data["symbol_path"]),
            locator=dict(data["locator"]),
            tags=tags,
        )


@dataclass
class StructureReport:
    """Compact, deterministic structure report for one source.

    Serializes to YAML with stable keys and stable item ordering. No
    timestamps, no op ids, no absolute paths, no environment-dependent
    parser metadata, no raw values or bodies.
    """

    source_id: str
    source_class: str
    locator_type: str
    content_hash: str
    language: str
    items: list[StructureItem] = field(default_factory=list)
    version: str = REPORT_VERSION

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_id": self.source_id,
            "source_class": self.source_class,
            "locator_type": self.locator_type,
            "content_hash": self.content_hash,
            "language": self.language,
            "items": [item.to_mapping() for item in self.items],
        }


def extract_structure(
    source_text: str,
    *,
    source_id: str,
    source_class: str,
    locator_type: str,
    raw_path: str,
    content_hash: str,
) -> StructureReport:
    """Build a :class:`StructureReport` for ``source_text``.

    Dispatches on ``source_class``. Raises
    :class:`StructureExtractError` for unsupported classes or
    malformed sources. Callers should treat the error as a
    batch-atomic refusal: the caller writes no report on failure.
    """
    if locator_type != "code_v1":
        raise StructureExtractError(
            f"structure_extract requires locator_type 'code_v1'; "
            f"got {locator_type!r}"
        )
    if source_class == "structured_yaml":
        items = _extract_yaml_structure(source_text, raw_path=raw_path)
        language = "yaml"
    elif source_class == "code":
        items, language = _extract_code_structure(
            source_text, raw_path=raw_path
        )
    else:
        raise StructureExtractError(
            f"unsupported source_class for structure_extract: {source_class!r}; "
            f"allowed: {sorted(SUPPORTED_SOURCE_CLASSES)}"
        )
    return StructureReport(
        source_id=source_id,
        source_class=source_class,
        locator_type=locator_type,
        content_hash=content_hash,
        language=language,
        items=items,
    )


def write_structure_report(workspace: Workspace, report: StructureReport) -> Path:
    """Atomically write ``report`` under ``state/structure/``.

    Uses the standard temp-file-and-rename pattern; the previous
    report (if any) stays in place until the replacement write
    succeeds.
    """
    target = workspace.structure_report_path(report.source_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    text = yaml.safe_dump(
        report.to_mapping(), sort_keys=False, allow_unicode=True
    )
    try:
        tmp.write_text(text, encoding="utf-8")
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    tmp.replace(target)
    return target


# ---- YAML extractor -----------------------------------------------------


def _extract_yaml_structure(source_text: str, *, raw_path: str) -> list[StructureItem]:
    """Walk the YAML node tree and emit one item per mapping key.

    Only mapping keys are emitted; scalar values, comments, and
    multiline blocks are deliberately not captured. Sequence items
    appear only as an indexed step in a nested key's ``symbol_path``
    — their scalar content is never stored.
    """
    try:
        root = yaml.compose(source_text)
    except yaml.YAMLError as exc:
        raise StructureExtractError(f"YAML parse error: {exc}") from exc
    if root is None:
        return []
    items: list[StructureItem] = []
    _walk_yaml_node(root, prefix=[], items=items, raw_path=raw_path)
    return items


def _walk_yaml_node(
    node: Any,
    *,
    prefix: list[str],
    items: list[StructureItem],
    raw_path: str,
) -> None:
    node_tag = type(node).__name__
    if node_tag == "MappingNode":
        for key_node, value_node in node.value:
            key_text = _yaml_scalar_key_text(key_node)
            if key_text is None:
                continue
            symbol_path = ".".join(prefix + [key_text])
            items.append(
                StructureItem(
                    kind="mapping_key",
                    name=key_text,
                    symbol_path=symbol_path,
                    locator=_yaml_locator(key_node, raw_path=raw_path),
                )
            )
            _walk_yaml_node(
                value_node,
                prefix=prefix + [key_text],
                items=items,
                raw_path=raw_path,
            )
    elif node_tag == "SequenceNode":
        for idx, child in enumerate(node.value):
            _walk_yaml_node(
                child,
                prefix=prefix + [f"[{idx}]"],
                items=items,
                raw_path=raw_path,
            )
    # ScalarNode values are intentionally not recorded.


def _yaml_scalar_key_text(node: Any) -> str | None:
    """Return a printable key name for a scalar key node; ``None``
    for non-scalar (mapping/sequence) keys, which we skip."""
    if type(node).__name__ != "ScalarNode":
        return None
    value = node.value
    return value if isinstance(value, str) else str(value)


def _yaml_locator(node: Any, *, raw_path: str) -> dict[str, Any]:
    """Build a ``code_v1`` locator from a YAML node's marks.

    PyYAML marks are 0-based; the locator is 1-based inclusive on
    both ends. ``end_col`` is clamped to >= ``start_col`` on the same
    line because PyYAML sometimes reports an end mark at the start
    of the next line (column 0).
    """
    start = node.start_mark
    end = node.end_mark
    start_line = start.line + 1
    start_col = start.column + 1
    end_line = end.line + 1
    end_col = end.column
    if end_col < 1:
        end_col = 1
    if end_line == start_line and end_col < start_col:
        end_col = start_col
    return {
        "locator_type": "code_v1",
        "path": raw_path,
        "start_line": start_line,
        "start_col": start_col,
        "end_line": end_line,
        "end_col": end_col,
    }


# ---- code extractor (tree-sitter, optional) -----------------------------


# Suffix-to-language dispatch for ``source_class="code"``. Each entry
# names the language key recorded on the report (``language``), the
# lazy loader that returns a ready ``(parser, language)`` pair, and the
# node walker that emits items for that language. Loaders raise
# ``StructureExtractError`` whose message names ``llloom[structured]``
# when the optional extra is not installed.

_CODE_SUFFIX_LANGUAGES = (".py", ".go", ".rs", ".ts", ".cs", ".java")


def _extract_code_structure(
    source_text: str, *, raw_path: str
) -> tuple[list[StructureItem], str]:
    """Dispatch by ``raw_path`` suffix to a language-specific walker.

    Returns ``(items, language)``. Unsupported suffixes raise
    :class:`StructureExtractError` with a clear message naming the
    supported suffixes and the install extra; this is the same
    failure mode callers already handle for an unsupported source
    class or wrong locator type.
    """
    suffix = _code_suffix(raw_path)
    if suffix == ".py":
        parser, _lang = _load_python_parser()
        return (
            _walk_code(parser, source_text, raw_path=raw_path, walker=_python_node_kind),
            "python",
        )
    if suffix == ".go":
        parser, _lang = _load_go_parser()
        return (
            _walk_code(parser, source_text, raw_path=raw_path, walker=_go_node_kind),
            "go",
        )
    if suffix == ".rs":
        parser, _lang = _load_rust_parser()
        return (
            _walk_code(parser, source_text, raw_path=raw_path, walker=_rust_node_kind),
            "rust",
        )
    if suffix == ".ts":
        parser, _lang = _load_typescript_parser()
        return (
            _walk_code(parser, source_text, raw_path=raw_path, walker=_typescript_node_kind),
            "typescript",
        )
    if suffix == ".cs":
        parser, _lang = _load_csharp_parser()
        return (
            _walk_code(parser, source_text, raw_path=raw_path, walker=_csharp_node_kind),
            "csharp",
        )
    if suffix == ".java":
        parser, _lang = _load_java_parser()
        return (
            _walk_code(parser, source_text, raw_path=raw_path, walker=_java_node_kind),
            "java",
        )
    raise StructureExtractError(
        f"unsupported source_class='code' file suffix {suffix!r} for "
        f"{raw_path!r}; supported suffixes are "
        f"{list(_CODE_SUFFIX_LANGUAGES)} (install with: "
        f"pip install \"llloom[structured]\")"
    )


def _code_suffix(raw_path: str) -> str:
    """Return the lowercase suffix of ``raw_path``, or ``''``."""
    idx = raw_path.rfind(".")
    if idx == -1 or idx == len(raw_path) - 1:
        return ""
    suffix = raw_path[idx:].lower()
    return suffix


# ---- generic walker shared across languages ----------------------------


def _walk_code(
    parser: Any,
    source_text: str,
    *,
    raw_path: str,
    walker,
) -> list[StructureItem]:
    """Parse ``source_text`` with ``parser`` and walk the tree.

    ``walker`` is a language-specific callable mapping a tree-sitter
    node to ``(kind, name, nested_prefix) | None``, optionally
    consulting an ``ancestor_types`` tuple for parent context (Rust
    uses this to recognise ``function_item`` inside ``impl_item`` and
    re-tag it as ``method``). Items are appended in document order,
    then sorted deterministically by (start_line, start_col,
    symbol_path) to pin ordering across tree-sitter versions.
    """
    tree = parser.parse(source_text.encode("utf-8"))
    root = tree.root_node
    items: list[StructureItem] = []
    _walk_node(
        root,
        source=source_text,
        prefix=[],
        items=items,
        raw_path=raw_path,
        walker=walker,
        ancestor_types=(),
    )
    items.sort(
        key=lambda i: (
            int(i.locator["start_line"]),
            int(i.locator["start_col"]),
            i.symbol_path,
        )
    )
    return items


def _walk_node(
    node: Any,
    *,
    source: str,
    prefix: list[str],
    items: list[StructureItem],
    raw_path: str,
    walker,
    ancestor_types: tuple[str, ...],
) -> None:
    """Recursive depth-first walk.

    Calls ``walker(node, source=..., ancestor_types=...)`` to decide
    whether this node contributes an item. A walker returning
    ``(kind, name, nested_prefix)`` extends ``prefix`` by
    ``nested_prefix`` for the recursion into this node's children —
    most languages just append the symbol's own name, but Rust
    ``impl`` blocks extend the prefix by the impl's target type
    without emitting an item themselves.
    """
    decision = walker(node, source=source, ancestor_types=ancestor_types)
    next_prefix = list(prefix)
    if decision is not None:
        # Walkers may return the 3-tuple (kind, name, nested_prefix)
        # or the 4-tuple (kind, name, nested_prefix, tags). The 4-tuple
        # form is used by the C# walker's Unity classifier; every
        # other walker stays on the 3-tuple shape and `tags` defaults
        # to () (empty), matching the `StructureItem.tags` default.
        if len(decision) == 4:
            kind, name, nested_prefix, tags = decision
        else:
            kind, name, nested_prefix = decision
            tags = ()
        if kind is not None and name:
            symbol_path = ".".join(prefix + [name])
            items.append(
                StructureItem(
                    kind=kind,
                    name=name,
                    symbol_path=symbol_path,
                    locator=_ts_locator(node, raw_path=raw_path),
                    tags=tuple(tags),
                )
            )
            next_prefix = prefix + [name]
        elif nested_prefix:
            # Container node (e.g. Rust `impl Type { ... }`) that adds
            # to the prefix without contributing an item.
            next_prefix = prefix + list(nested_prefix)
    child_ancestors = ancestor_types + (getattr(node, "type", ""),)
    for child in getattr(node, "children", []):
        _walk_node(
            child,
            source=source,
            prefix=next_prefix,
            items=items,
            raw_path=raw_path,
            walker=walker,
            ancestor_types=child_ancestors,
        )


def _identifier_text(node: Any, source: str) -> str | None:
    """Return the source text under ``node`` (used for identifier
    nodes). Returns ``None`` if the node lacks byte offsets."""
    start_byte = getattr(node, "start_byte", None)
    end_byte = getattr(node, "end_byte", None)
    if start_byte is None or end_byte is None:
        return None
    encoded = source.encode("utf-8")
    return encoded[start_byte:end_byte].decode("utf-8", errors="replace")


def _named_field_or_identifier(node: Any, source: str) -> str | None:
    """Common pattern: read the ``name`` field if present, otherwise
    scan children for the first ``identifier``-shaped node."""
    try:
        name_node = node.child_by_field_name("name")
    except Exception:
        name_node = None
    if name_node is None:
        for child in getattr(node, "children", []):
            ctype = getattr(child, "type", "")
            if ctype in {"identifier", "type_identifier", "property_identifier"}:
                name_node = child
                break
    if name_node is None:
        return None
    return _identifier_text(name_node, source)


# ---- Python ------------------------------------------------------------


def _python_node_kind(node: Any, *, source: str, ancestor_types: tuple[str, ...]):
    _ = ancestor_types  # Python walker has no parent-context dependence.
    ts_type = getattr(node, "type", "")
    if ts_type == "class_definition":
        kind = "class"
    elif ts_type == "function_definition":
        kind = "function"
    elif ts_type == "async_function_definition":
        kind = "async_function"
    else:
        return None
    name = _named_field_or_identifier(node, source)
    if name is None:
        return None
    return (kind, name, [name])


def _load_python_parser():
    try:
        from tree_sitter import Parser  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter is required for source_class='code' structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    try:
        from tree_sitter_python import language as _py_language  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter-python is required for Python (.py) structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    return _build_parser(Parser, _py_language())


# ---- Go ---------------------------------------------------------------


def _go_node_kind(node: Any, *, source: str, ancestor_types: tuple[str, ...]):
    _ = ancestor_types  # Go walker has no parent-context dependence.
    ts_type = getattr(node, "type", "")
    if ts_type == "function_declaration":
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        return ("function", name, [name])
    if ts_type == "method_declaration":
        # method_declaration carries a ``receiver`` field; methods are
        # recorded under ``<ReceiverType>.<MethodName>`` to match the
        # symbol-path convention used by Rust impl methods and
        # TypeScript class methods.
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        receiver_type = _go_method_receiver_type(node, source)
        if receiver_type:
            qualified = f"{receiver_type}.{name}"
            return ("method", qualified, [qualified])
        return ("method", name, [name])
    if ts_type == "type_spec":
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        return ("type", name, [name])
    return None


def _go_method_receiver_type(node: Any, source: str) -> str | None:
    """Extract the receiver-type identifier from a Go ``method_declaration``.

    A Go method declaration has the shape
    ``func (r *Receiver) Name(args) { ... }``. tree-sitter's Go
    grammar exposes the receiver as a ``parameter_list`` child whose
    inner ``parameter_declaration`` carries a ``type`` field — that
    type may be an ``identifier`` (value receiver) or a
    ``pointer_type`` wrapping an ``identifier``. We walk down to the
    first ``identifier`` / ``type_identifier`` and return its text.
    """
    receiver = None
    try:
        receiver = node.child_by_field_name("receiver")
    except Exception:
        receiver = None
    if receiver is None:
        return None
    return _first_type_identifier(receiver, source)


def _first_type_identifier(node: Any, source: str) -> str | None:
    """Depth-first search for the first ``identifier`` or
    ``type_identifier`` descendant; used for Go receivers and Rust
    impl targets where the type name is nested under a wrapper."""
    stack = [node]
    while stack:
        cur = stack.pop()
        ctype = getattr(cur, "type", "")
        if ctype in {"type_identifier", "identifier"}:
            text = _identifier_text(cur, source)
            if text:
                return text
        stack.extend(reversed(list(getattr(cur, "children", []))))
    return None


def _load_go_parser():
    try:
        from tree_sitter import Parser  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter is required for source_class='code' structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    try:
        from tree_sitter_go import language as _go_language  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter-go is required for Go (.go) structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    return _build_parser(Parser, _go_language())


# ---- Rust -------------------------------------------------------------


def _rust_node_kind(node: Any, *, source: str, ancestor_types: tuple[str, ...]):
    ts_type = getattr(node, "type", "")
    if ts_type == "function_item":
        # function_item appears both at module level and inside
        # impl_item. When the walker is currently inside an impl_item
        # subtree, re-tag the item as a "method"; the prefix has
        # already been extended by the impl target type via the
        # impl_item branch below.
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        kind = "method" if "impl_item" in ancestor_types else "function"
        return (kind, name, [name])
    if ts_type == "struct_item":
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        return ("struct", name, [name])
    if ts_type == "enum_item":
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        return ("enum", name, [name])
    if ts_type == "trait_item":
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        return ("trait", name, [name])
    if ts_type == "impl_item":
        # An `impl Type { ... }` block contributes no item of its own.
        # The walker recurses into its children with the prefix
        # extended by the impl target type, so any `function_item`
        # inside becomes "<Type>.<fn>" with kind "method" — we rewrite
        # the kind below via the second walker stage.
        target = _rust_impl_target(node, source)
        if not target:
            return None
        return (None, None, [target])
    return None


def _rust_impl_target(node: Any, source: str) -> str | None:
    """Return the target-type identifier of a Rust ``impl_item``.

    tree-sitter's Rust grammar exposes the implemented type as the
    ``type`` field. We resolve the first identifier underneath that
    field; for trait impls (``impl Trait for Type``) tree-sitter
    distinguishes ``trait`` and ``type`` fields and we prefer the
    ``type`` field — which is the inherent target.
    """
    try:
        type_field = node.child_by_field_name("type")
    except Exception:
        type_field = None
    if type_field is None:
        return None
    return _first_type_identifier(type_field, source)


def _load_rust_parser():
    try:
        from tree_sitter import Parser  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter is required for source_class='code' structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    try:
        from tree_sitter_rust import language as _rust_language  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter-rust is required for Rust (.rs) structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    return _build_parser(Parser, _rust_language())


# ---- TypeScript -------------------------------------------------------


def _typescript_node_kind(node: Any, *, source: str, ancestor_types: tuple[str, ...]):
    _ = ancestor_types  # TS walker has no parent-context dependence.
    ts_type = getattr(node, "type", "")
    if ts_type == "function_declaration":
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        return ("function", name, [name])
    if ts_type == "class_declaration":
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        return ("class", name, [name])
    if ts_type == "interface_declaration":
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        return ("interface", name, [name])
    if ts_type == "type_alias_declaration":
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        return ("type_alias", name, [name])
    if ts_type == "method_definition":
        name = _named_field_or_identifier(node, source)
        if name is None:
            return None
        return ("method", name, [name])
    return None


def _load_typescript_parser():
    try:
        from tree_sitter import Parser  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter is required for source_class='code' structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    try:
        # tree-sitter-typescript exposes two grammars (``typescript``
        # and ``tsx``) via the same package. We use the ``typescript``
        # grammar here; ``.tsx`` files are deliberately out of scope
        # for this slice.
        from tree_sitter_typescript import language_typescript as _ts_language  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter-typescript is required for TypeScript (.ts) structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    return _build_parser(Parser, _ts_language())


# ---- C# ---------------------------------------------------------------


def _csharp_node_kind(node: Any, *, source: str, ancestor_types: tuple[str, ...]):
    """Tree-sitter-c-sharp node-kind adapter.

    Admits the narrow C# declaration set: class / interface / struct
    / enum / method. Nested declarations inside a class / interface /
    struct / enum naturally inherit their container's prefix via the
    generic walker, so a method inside ``class Store`` gets
    ``symbol_path == "Store.MethodName"``. Namespaces, fields,
    properties, events, attributes, and arbitrary body declarations
    are out of scope.

    Unity bridge v1: a ``class_declaration`` that **directly**
    inherits from ``MonoBehaviour`` (or a qualified name ending in
    ``.MonoBehaviour``) is re-tagged ``kind == "unity_component"``
    and additionally carries the generic framework tags
    ``("framework:unity", "role:component")`` on the emitted
    ``StructureItem``. The ``kind`` rename is preserved for
    backward compatibility; the tags are the generic metadata
    channel future framework classifiers will reuse. No transitive
    inheritance, no alias / using-graph resolution, no lifecycle /
    scene / prefab / asmdef awareness. ``ScriptableObject`` and
    every other Unity engine type remain deferred.
    """
    _ = ancestor_types  # C# walker has no parent-context dependence.
    ts_type = getattr(node, "type", "")
    tags: tuple[str, ...] = ()
    if ts_type == "class_declaration":
        if _is_direct_monobehaviour_base(node, source):
            kind = "unity_component"
            tags = _UNITY_COMPONENT_TAGS
        else:
            kind = "class"
    elif ts_type == "interface_declaration":
        kind = "interface"
    elif ts_type == "struct_declaration":
        kind = "struct"
    elif ts_type == "enum_declaration":
        kind = "enum"
    elif ts_type == "method_declaration":
        kind = "method"
    else:
        return None
    name = _named_field_or_identifier(node, source)
    if name is None:
        return None
    return (kind, name, [name], tags)


# Generic framework-tagging set for direct C# MonoBehaviour subclasses.
# Order is fixed so serialized reports stay byte-deterministic. The
# tags follow the lowercase ASCII ``prefix:value`` convention shared
# across future framework classifiers.
_UNITY_COMPONENT_TAGS: tuple[str, ...] = ("framework:unity", "role:component")


def _is_direct_monobehaviour_base(node: Any, source: str) -> bool:
    """Return True iff a C# class declaration directly inherits from
    ``MonoBehaviour`` or a qualified form ending in
    ``.MonoBehaviour``.

    The check is intentionally textual and shallow: it reads the
    bytes of every immediate entry in the class's ``base_list``
    (located via the ``bases`` field on tree-sitter-c-sharp; or a
    fallback scan of the node's direct children for a ``base_list``
    child to stay robust against grammar-binding minor variations)
    and compares against ``MonoBehaviour`` / ``*.MonoBehaviour``.
    No transitive base-graph walk, no alias resolution, no
    `using`-graph reasoning. Anything other than a direct base
    match returns False.
    """
    base_list = None
    try:
        base_list = node.child_by_field_name("bases")
    except Exception:
        base_list = None
    if base_list is None:
        for child in getattr(node, "children", []):
            if getattr(child, "type", "") == "base_list":
                base_list = child
                break
    if base_list is None:
        return False
    for entry in getattr(base_list, "children", []):
        entry_type = getattr(entry, "type", "")
        if entry_type not in {"identifier", "qualified_name", "generic_name"}:
            continue
        text = _identifier_text(entry, source)
        if not text:
            continue
        stripped = text.strip()
        if stripped == "MonoBehaviour" or stripped.endswith(".MonoBehaviour"):
            return True
    return False


def _load_csharp_parser():
    try:
        from tree_sitter import Parser  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter is required for source_class='code' structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    try:
        from tree_sitter_c_sharp import language as _cs_language  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter-c-sharp is required for C# (.cs) structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    return _build_parser(Parser, _cs_language())


# ---- Java -------------------------------------------------------------


def _java_node_kind(node: Any, *, source: str, ancestor_types: tuple[str, ...]):
    """Tree-sitter-java node-kind adapter (Slice 082 V1).

    Admits the narrow Java declaration set the dev note prioritises:
    ``class`` / ``interface`` / ``enum`` / ``record`` / ``method`` /
    ``constructor`` / ``field``. Nested declarations inherit their
    container's prefix naturally via the generic walker, so a method
    inside ``class Project`` surfaces as
    ``symbol_path == "Project.evaluate"``.

    ``field_declaration`` in tree-sitter-java wraps one or more
    ``variable_declarator`` children (because a single declaration
    can list several variables, e.g. ``int a, b;``). **V1 emits
    exactly one ``field`` item per ``field_declaration`` node — the
    first ``variable_declarator``'s name only.** Multi-declarator
    forms like ``int a, b;`` therefore record `a` and drop `b`.
    Multi-declarator coverage is a deliberate non-goal for V1; the
    extension point lives in :func:`_java_field_declaration_items`,
    not in the generic walker. Local variable declarations inside
    a method body are NOT field declarations and stay out of the
    walker — only mappings keyed on ``field_declaration`` are
    considered. Annotations, modifiers, comments, throws clauses,
    parameter names, scalar literals, statement bodies, and import
    declarations are deliberately out of scope for V1.

    The walker has no parent-context dependence beyond the generic
    prefix machinery; ``ancestor_types`` is accepted for signature
    compatibility but not consulted.
    """
    _ = ancestor_types
    ts_type = getattr(node, "type", "")
    if ts_type == "class_declaration":
        kind = "class"
    elif ts_type == "interface_declaration":
        kind = "interface"
    elif ts_type == "enum_declaration":
        kind = "enum"
    elif ts_type == "record_declaration":
        kind = "record"
    elif ts_type == "method_declaration":
        kind = "method"
    elif ts_type == "constructor_declaration":
        # tree-sitter-java's ``constructor_declaration`` carries a
        # ``name`` field whose identifier text is the constructor
        # name (== the enclosing class name). The generic
        # ``_named_field_or_identifier`` resolver picks that up, and
        # the enclosing class's prefix gives the canonical
        # ``Project.Project`` symbol path the dev note documents.
        kind = "constructor"
    elif ts_type == "field_declaration":
        return _java_field_declaration_items(node, source)
    else:
        return None
    name = _named_field_or_identifier(node, source)
    if name is None:
        return None
    return (kind, name, [name])


def _java_field_declaration_items(node: Any, source: str):
    """Return the first variable declarator under a Java
    ``field_declaration`` so the generic walker emits exactly one
    ``field`` item per declaration.

    **V1 behavior (Slice 082):** Java ``field_declaration`` nodes
    can legitimately declare multiple variables in one statement
    (``int a, b;``). V1 records only the first
    ``variable_declarator`` whose ``_named_field_or_identifier``
    resolves to a non-None name. The second and any subsequent
    declarators are deliberately dropped — the walker contract
    stays simple and deterministic, and the resulting item count
    matches Java IDE "Outline" panels for the common
    one-variable-per-field idiom.

    **Future multi-declarator coverage extension point:** any
    future slice that adds multi-declarator support extends this
    function (e.g. to return a list of items, or to call the
    generic walker explicitly per declarator). The generic
    `_walk_node` machinery should NOT be widened to special-case
    Java multi-declarator declarations; the asymmetry belongs
    here.
    """
    for child in getattr(node, "children", []):
        ctype = getattr(child, "type", "")
        if ctype == "variable_declarator":
            name = _named_field_or_identifier(child, source)
            if name is None:
                continue
            return ("field", name, [name])
    return None


def _load_java_parser():
    try:
        from tree_sitter import Parser  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter is required for source_class='code' structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    try:
        from tree_sitter_java import language as _java_language  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StructureExtractError(
            "tree-sitter-java is required for Java (.java) structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc
    return _build_parser(Parser, _java_language())


# ---- shared tree-sitter parser builder + locator helper ---------------


def _build_parser(parser_cls, language_obj):
    """Construct a tree-sitter ``Parser`` and bind the language.

    Different combinations of `tree-sitter` core and per-language
    grammar packages disagree on what a grammar object actually is:

    - older grammar packages return a `tree_sitter.Language` already
      and either `parser.language = lang` or
      `parser.set_language(lang)` works directly;
    - newer grammar packages (e.g. current `tree_sitter_c_sharp`)
      return a raw PyCapsule from `language()`; current `tree-sitter`
      then rejects the capsule and demands a `tree_sitter.Language`
      wrapper.

    Walk a defensive candidate sequence so the same shared helper
    covers every per-language loader (Python, Go, Rust, TypeScript,
    C#, and any future tree-sitter-backed language). The first
    successful binding wins; if every candidate fails, raise a
    :class:`StructureExtractError` with a clear message and the
    `llloom[structured]` install hint so users have a single
    actionable next step.
    """
    parser = parser_cls()
    bound = _bind_tree_sitter_language(parser, language_obj)
    return parser, bound


def _bind_tree_sitter_language(parser: Any, language_obj: Any) -> Any:
    """Bind ``language_obj`` to ``parser`` defensively.

    Lazy-imports ``tree_sitter.Language`` (the optional extra is
    already in scope by the time any caller of this helper runs).
    Tries the candidate sequence:

    1. direct ``parser.language = language_obj``;
    2. wrap with ``Language(language_obj)`` then assign — handles
       PyCapsule-style grammar returns;
    3. older ``parser.set_language(language_obj)`` API;
    4. older ``parser.set_language(Language(language_obj))``.

    Returns the language object that was actually bound (either the
    original or its wrapped form) so callers can keep a reference for
    later use without re-detecting.
    """
    try:
        from tree_sitter import Language  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — guarded by callers
        raise StructureExtractError(
            "tree-sitter is required for source_class='code' structure_extract; "
            "install the optional extra with: pip install \"llloom[structured]\""
        ) from exc

    failures: list[str] = []

    # Build the unique candidate sequence: original first, wrapped
    # second. If `Language(...)` itself raises (e.g. the input is
    # already a `Language` and the wrapper refuses re-wrapping), skip
    # that candidate rather than abort the whole sequence.
    candidates: list[Any] = [language_obj]
    if not isinstance(language_obj, Language):
        try:
            candidates.append(Language(language_obj))
        except Exception as exc:
            failures.append(f"Language(grammar) wrap failed: {exc!r}")

    for candidate in candidates:
        # Path A: attribute assignment (tree-sitter 0.22+).
        try:
            parser.language = candidate
            return candidate
        except Exception as exc:
            failures.append(f"parser.language = candidate failed: {exc!r}")
        # Path B: legacy setter (tree-sitter 0.21).
        setter = getattr(parser, "set_language", None)
        if callable(setter):
            try:
                setter(candidate)
                return candidate
            except Exception as exc:
                failures.append(f"parser.set_language(candidate) failed: {exc!r}")

    raise StructureExtractError(
        "could not bind tree-sitter grammar to parser; the installed "
        "tree-sitter / grammar combination is not supported by llloom. "
        "Ensure the structured extra is current: "
        "pip install -U \"llloom[structured]\". "
        "(attempts: " + "; ".join(failures) + ")"
    )


def _ts_locator(node: Any, *, raw_path: str) -> dict[str, Any]:
    """Build a 1-based inclusive ``code_v1`` locator from a
    tree-sitter node's (row, column) marks."""
    start_row, start_col = node.start_point
    end_row, end_col = node.end_point
    start_line = start_row + 1
    s_col = start_col + 1
    end_line_1 = end_row + 1
    e_col = end_col
    if e_col < 1:
        e_col = 1
    if end_line_1 == start_line and e_col < s_col:
        e_col = s_col
    return {
        "locator_type": "code_v1",
        "path": raw_path,
        "start_line": start_line,
        "start_col": s_col,
        "end_line": end_line_1,
        "end_col": e_col,
    }
