"""Contract tests for the `structure_extract` ingest integration.

The ingest path must register the source, write a deterministic
derived structure report under ``state/structure/<source_id>.yaml``,
and return **without** invoking ``LLMInvoke``. Reports must not leak
raw values, comments, or bodies. Malformed sources refuse cleanly
with no partial report written.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml as yaml_mod

from llloom.cli import main as cli_main
from llloom.llm.harness import LLMInvoke
from llloom.ops.ingest import ingest
from llloom.workspace.layout import Workspace


YAML_SOURCE = (
    "policies:\n"
    "  markdown_prose: claim_extract_and_view_render\n"
    "  legal_act: claim_extract\n"
    "defaults:\n"
    "  unknown: deny\n"
)


def _seed_yaml_source(tmp_path: Path) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "policies.yaml"
    src.write_text(YAML_SOURCE, encoding="utf-8")
    return ws, src


# --- 1. structure_extract ingest writes a report, no claims/pages --------


def test_structure_extract_writes_report_and_no_claims(tmp_path: Path) -> None:
    ws, src = _seed_yaml_source(tmp_path)
    result = ingest(
        ws,
        src,
        source_id="src.policies",
        source_class="structured_yaml",
    )
    assert result.succeeded
    assert result.policy == "structure_extract"
    assert result.structure_reports == ["state/structure/src.policies.yaml"]
    assert result.claims_created == []
    assert result.pages_rendered == []
    assert result.merge_proposals_created == []
    report_path = ws.structure_report_path("src.policies")
    assert report_path.is_file()
    loaded = yaml_mod.safe_load(report_path.read_text(encoding="utf-8"))
    assert loaded["version"] == "structure_report_v1"
    assert loaded["source_id"] == "src.policies"
    assert loaded["source_class"] == "structured_yaml"
    assert loaded["language"] == "yaml"
    symbol_paths = {item["symbol_path"] for item in loaded["items"]}
    assert "policies" in symbol_paths
    assert "policies.markdown_prose" in symbol_paths
    assert "defaults.unknown" in symbol_paths


# --- 2. structure_extract never invokes LLMInvoke -----------------------


def test_structure_extract_does_not_invoke_llm_invoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_yaml_source(tmp_path)

    def _fail(self, **kwargs):  # noqa: ANN001 - test stub
        raise AssertionError(
            "structure_extract ingest must not invoke LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r}"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _fail)

    result = ingest(
        ws,
        src,
        source_id="src.policies",
        source_class="structured_yaml",
    )
    assert result.succeeded
    assert result.structure_reports == ["state/structure/src.policies.yaml"]


# --- 3. index_only remains unchanged; no structure report ---------------


def _wire_index_only(ws: Workspace) -> None:
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  sensitive:\n"
        "    locator: markdown_prose_v1\n",
        encoding="utf-8",
    )
    (ws.schema / "ingest_policies.yaml").write_text(
        "policies:\n"
        "  markdown_prose: claim_extract_and_view_render\n"
        "  sensitive: index_only\n"
        "defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )


def test_index_only_ingest_writes_no_structure_report(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    _wire_index_only(ws)
    src = ws.raw_sources / "contract.md"
    src.write_text(
        "# Contract\n\nNet-30 with a 2% early-payment discount.\n",
        encoding="utf-8",
    )
    result = ingest(ws, src, source_id="src.contract", source_class="sensitive")
    assert result.succeeded
    assert result.policy == "index_only"
    assert result.structure_reports == []
    assert not ws.state_structure.exists() or not list(ws.state_structure.iterdir())


# --- 4. malformed YAML refuses cleanly, no partial report ---------------


def test_structure_extract_malformed_yaml_refuses(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "broken.yaml"
    src.write_text("policies:\n  unterminated: [abc\n", encoding="utf-8")
    result = ingest(
        ws,
        src,
        source_id="src.broken",
        source_class="structured_yaml",
    )
    assert not result.succeeded
    assert result.refusal_reason is not None
    assert "structure_extract failed" in result.refusal_reason
    # No partial report, no stray tmp file.
    assert not ws.structure_report_path("src.broken").exists()
    tmp = ws.structure_report_path("src.broken").with_suffix(".yaml.tmp")
    assert not tmp.exists()


# --- 5. CLI ingest emits JSON containing the report path ----------------


def test_cli_structure_extract_emits_report_path(tmp_path: Path) -> None:
    ws, src = _seed_yaml_source(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main([
            "--root", str(ws.root),
            "ingest", str(src),
            "--source-id", "src.policies",
            "--source-class", "structured_yaml",
        ])
    assert rc == 0, buf.getvalue()
    payload = json.loads(buf.getvalue())
    assert payload["policy"] == "structure_extract"
    assert payload["structure_reports"] == ["state/structure/src.policies.yaml"]
    assert payload["claims_created"] == []


# --- 6. reports do not contain seeded canary text -----------------------


def test_structure_report_does_not_contain_scalar_values_or_comments(
    tmp_path: Path,
) -> None:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "poisoned.yaml"
    # Plant recognizable strings inside a scalar value and a comment.
    # The report must store neither.
    src.write_text(
        "# POISON_COMMENT_marker_xyz\n"
        "policies:\n"
        "  one: POISON_VALUE_marker_abc\n",
        encoding="utf-8",
    )
    ingest(
        ws,
        src,
        source_id="src.poisoned",
        source_class="structured_yaml",
    )
    report_text = ws.structure_report_path("src.poisoned").read_text(encoding="utf-8")
    assert "POISON_COMMENT_marker_xyz" not in report_text
    assert "POISON_VALUE_marker_abc" not in report_text


# --- 7. repeated extraction on unchanged input is byte-identical --------


def test_repeated_structure_extract_is_byte_identical(tmp_path: Path) -> None:
    ws, src = _seed_yaml_source(tmp_path)
    ingest(ws, src, source_id="src.policies", source_class="structured_yaml")
    first = ws.structure_report_path("src.policies").read_text(encoding="utf-8")
    ingest(ws, src, source_id="src.policies", source_class="structured_yaml")
    second = ws.structure_report_path("src.policies").read_text(encoding="utf-8")
    assert first == second


# --- Broader-language `code` slice ----------------------------------------
#
# The default suite must run offline without ``tree-sitter-go``,
# ``tree-sitter-rust``, or ``tree-sitter-typescript`` installed. Each
# contract test monkeypatches the per-language loader so the ingest
# path executes end-to-end without requiring those wheels.


class _FakeNode:
    """Tree-sitter node mock — minimal subset used by the walker."""

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
    def __init__(self, root: _FakeNode) -> None:
        self._root = root

    def parse(self, _data: bytes):
        return _FakeTree(self._root)


def _named(ts_type: str, name: str, *, source: str, identifier_ts_type: str = "identifier") -> _FakeNode:
    """Build a fake node whose ``name`` field points at ``name`` in
    ``source``."""
    off = source.index(name)
    name_node = _FakeNode(identifier_ts_type, start_byte=off, end_byte=off + len(name))
    return _FakeNode(ts_type, children=[name_node], fields={"name": name_node})


def _go_root(source: str) -> _FakeNode:
    return _FakeNode(
        "source_file",
        children=[_named("function_declaration", "Main", source=source)],
    )


def _rust_root(source: str) -> _FakeNode:
    return _FakeNode(
        "source_file",
        children=[_named("function_item", "main", source=source)],
    )


def _ts_root(source: str) -> _FakeNode:
    return _FakeNode(
        "program",
        children=[_named("function_declaration", "main", source=source)],
    )


def test_structure_extract_ingest_go_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llloom.structured.extract as ext

    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "main.go"
    src.write_text("package main\nfunc Main() {}\n", encoding="utf-8")

    monkeypatch.setattr(
        ext,
        "_load_go_parser",
        lambda: (_FakeParser(_go_root("package main\nfunc Main() {}\n")), object()),
    )

    result = ingest(ws, src, source_id="src.go.main", source_class="code")
    assert result.succeeded
    assert result.policy == "structure_extract"
    assert result.structure_reports == ["state/structure/src.go.main.yaml"]
    assert result.claims_created == []
    assert result.pages_rendered == []
    report_path = ws.structure_report_path("src.go.main")
    loaded = yaml_mod.safe_load(report_path.read_text(encoding="utf-8"))
    assert loaded["version"] == "structure_report_v1"
    assert loaded["language"] == "go"
    assert loaded["source_class"] == "code"
    assert {it["symbol_path"] for it in loaded["items"]} == {"Main"}


def test_structure_extract_ingest_rust_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llloom.structured.extract as ext

    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "lib.rs"
    src.write_text("fn main() {}\n", encoding="utf-8")

    monkeypatch.setattr(
        ext,
        "_load_rust_parser",
        lambda: (_FakeParser(_rust_root("fn main() {}\n")), object()),
    )

    result = ingest(ws, src, source_id="src.rust.lib", source_class="code")
    assert result.succeeded
    assert result.policy == "structure_extract"
    assert result.structure_reports == ["state/structure/src.rust.lib.yaml"]
    loaded = yaml_mod.safe_load(
        ws.structure_report_path("src.rust.lib").read_text(encoding="utf-8")
    )
    assert loaded["language"] == "rust"
    assert {it["symbol_path"] for it in loaded["items"]} == {"main"}


def test_structure_extract_ingest_typescript_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llloom.structured.extract as ext

    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "index.ts"
    src.write_text("export function main(): void {}\n", encoding="utf-8")

    monkeypatch.setattr(
        ext,
        "_load_typescript_parser",
        lambda: (
            _FakeParser(_ts_root("export function main(): void {}\n")),
            object(),
        ),
    )

    result = ingest(ws, src, source_id="src.ts.index", source_class="code")
    assert result.succeeded
    assert result.policy == "structure_extract"
    assert result.structure_reports == ["state/structure/src.ts.index.yaml"]
    loaded = yaml_mod.safe_load(
        ws.structure_report_path("src.ts.index").read_text(encoding="utf-8")
    )
    assert loaded["language"] == "typescript"
    assert {it["symbol_path"] for it in loaded["items"]} == {"main"}


def test_broader_languages_never_invoke_llm_and_cli_shape_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three broader-language ingests still flow through the
    `structure_extract` policy cutoff, never invoking `LLMInvoke`,
    and the CLI's JSON shape is unchanged apart from the report
    path under the existing `structure_reports` field."""
    import llloom.structured.extract as ext

    ws = Workspace.init(tmp_path)
    cases = [
        ("rust", "lib.rs", "src.rust.x", "fn main() {}\n", _rust_root, "_load_rust_parser"),
        (
            "typescript",
            "index.ts",
            "src.ts.x",
            "export function main(): void {}\n",
            _ts_root,
            "_load_typescript_parser",
        ),
        (
            "go",
            "main.go",
            "src.go.x",
            "package main\nfunc Main() {}\n",
            _go_root,
            "_load_go_parser",
        ),
    ]

    def _fail_invoke(self, **kwargs):  # noqa: ANN001 - test stub
        raise AssertionError(
            "broader-language structure_extract must not invoke LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r}"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _fail_invoke)

    for language, filename, source_id, text, root_factory, loader_name in cases:
        src = ws.raw_sources / filename
        src.write_text(text, encoding="utf-8")
        monkeypatch.setattr(
            ext,
            loader_name,
            lambda root_factory=root_factory, text=text: (
                _FakeParser(root_factory(text)),
                object(),
            ),
        )
        result = ingest(ws, src, source_id=source_id, source_class="code")
        assert result.succeeded, result.refusal_reason
        assert result.structure_reports == [
            f"state/structure/{source_id}.yaml"
        ]

    # CLI smoke: the JSON shape is unchanged apart from the new
    # report under the existing `structure_reports` field.
    src = ws.raw_sources / "two.go"
    src.write_text("package main\nfunc Main() {}\n", encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main([
            "--root", str(ws.root),
            "ingest", str(src),
            "--source-id", "src.go.two",
            "--source-class", "code",
        ])
    assert rc == 0, buf.getvalue()
    payload = json.loads(buf.getvalue())
    assert payload["policy"] == "structure_extract"
    assert payload["structure_reports"] == ["state/structure/src.go.two.yaml"]
    assert payload["claims_created"] == []


def _csharp_root(source: str) -> _FakeNode:
    return _FakeNode(
        "compilation_unit",
        children=[_named("class_declaration", "Store", source=source)],
    )


def test_structure_extract_ingest_csharp_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llloom.structured.extract as ext

    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "Store.cs"
    src.write_text("class Store {}\n", encoding="utf-8")

    monkeypatch.setattr(
        ext,
        "_load_csharp_parser",
        lambda: (_FakeParser(_csharp_root("class Store {}\n")), object()),
    )

    result = ingest(ws, src, source_id="src.cs.store", source_class="code")
    assert result.succeeded
    assert result.policy == "structure_extract"
    assert result.structure_reports == ["state/structure/src.cs.store.yaml"]
    assert result.claims_created == []
    assert result.pages_rendered == []
    report_path = ws.structure_report_path("src.cs.store")
    loaded = yaml_mod.safe_load(report_path.read_text(encoding="utf-8"))
    assert loaded["version"] == "structure_report_v1"
    assert loaded["language"] == "csharp"
    assert loaded["source_class"] == "code"
    assert {it["symbol_path"] for it in loaded["items"]} == {"Store"}


def _csharp_unity_root(source: str) -> _FakeNode:
    """Build a fake compilation unit containing
    `class PlayerController : MonoBehaviour {}` — the structure
    extractor should re-tag this as `kind == "unity_component"`."""
    name_offset = source.index("PlayerController")
    name_node = _FakeNode(
        "identifier",
        start_byte=name_offset,
        end_byte=name_offset + len("PlayerController"),
    )
    base_offset = source.index("MonoBehaviour")
    base_entry = _FakeNode(
        "identifier",
        start_byte=base_offset,
        end_byte=base_offset + len("MonoBehaviour"),
    )
    base_list = _FakeNode("base_list", children=[base_entry])
    class_node = _FakeNode(
        "class_declaration",
        children=[name_node, base_list],
        fields={"name": name_node, "bases": base_list},
    )
    return _FakeNode("compilation_unit", children=[class_node])


def test_structure_extract_ingest_csharp_unity_component_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unity bridge v1: a `.cs` file whose class directly inherits
    from `MonoBehaviour` surfaces as `kind == "unity_component"` in
    the persisted structure report — no other downstream contract
    changes are required."""
    import llloom.structured.extract as ext

    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "PlayerController.cs"
    body = "class PlayerController : MonoBehaviour {}\n"
    src.write_text(body, encoding="utf-8")

    monkeypatch.setattr(
        ext,
        "_load_csharp_parser",
        lambda: (_FakeParser(_csharp_unity_root(body)), object()),
    )

    result = ingest(
        ws, src, source_id="src.cs.player", source_class="code"
    )
    assert result.succeeded
    assert result.structure_reports == ["state/structure/src.cs.player.yaml"]
    loaded = yaml_mod.safe_load(
        ws.structure_report_path("src.cs.player").read_text(encoding="utf-8")
    )
    assert loaded["language"] == "csharp"
    kinds = {(it["symbol_path"], it["kind"]) for it in loaded["items"]}
    assert kinds == {("PlayerController", "unity_component")}


# ---- Java (Slice 082) -------------------------------------------------


def _java_root(source: str) -> _FakeNode:
    """Build a fake Java compilation unit covering every V1 item kind:
    class + nested constructor + nested method + nested field +
    sibling interface + enum + record. Mirrors the unit-test fixture
    in ``tests/unit/test_structure_extract.py`` but slim enough for
    the ingest contract.
    """
    # Field declaration under the class: tree-sitter-java wraps the
    # declarator (the field-name carrier) under ``field_declaration``.
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
        start_point=(2, 4),
        end_point=(2, 24),
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
        start_point=(3, 4),
        end_point=(3, 30),
        children=[ctor_name],
        fields={"name": ctor_name},
    )

    method_decl = _named("method_declaration", "evaluate", source=source)
    method_decl.start_point = (4, 4)
    method_decl.end_point = (4, 40)

    class_decl = _named(
        "class_declaration",
        "Project",
        source=source,
    )
    class_decl.children = list(class_decl.children) + [
        field_decl,
        ctor_decl,
        method_decl,
    ]
    class_decl.start_point = (1, 0)
    class_decl.end_point = (5, 1)

    interface_decl = _named("interface_declaration", "Metric", source=source)
    interface_decl.start_point = (7, 0)
    interface_decl.end_point = (7, 18)

    enum_decl = _named("enum_declaration", "PatchKind", source=source)
    enum_decl.start_point = (9, 0)
    enum_decl.end_point = (9, 30)

    record_decl = _named("record_declaration", "PatchRecord", source=source)
    record_decl.start_point = (11, 0)
    record_decl.end_point = (11, 30)

    return _FakeNode(
        "program",
        children=[class_decl, interface_decl, enum_decl, record_decl],
    )


def test_structure_extract_ingest_java_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.java` file ingested under ``source_class="code"`` writes
    exactly one structure report; no `LLMInvoke` is invoked; no
    claim or page is written; the persisted report records
    ``language: java`` and the seven V1 Java symbol paths.
    """
    import llloom.structured.extract as ext

    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "Project.java"
    body = (
        "public class Project {\n"
        "    private int maxsize = 99;\n"
        "    public Project() {}\n"
        "    public int evaluate() { return 1; }\n"
        "}\n"
        "\n"
        "interface Metric {}\n"
        "\n"
        "enum PatchKind { LAND, WATER }\n"
        "\n"
        "record PatchRecord(int id) {}\n"
    )
    src.write_text(body, encoding="utf-8")

    monkeypatch.setattr(
        ext,
        "_load_java_parser",
        lambda: (_FakeParser(_java_root(body)), object()),
    )

    # Guard the model-free claim: any `LLMInvoke.invoke` call would
    # fail the test immediately, even though the structure_extract
    # policy never reaches the harness.
    from llloom.llm.harness import LLMInvoke

    monkeypatch.setattr(
        LLMInvoke,
        "invoke",
        lambda self, *a, **kw: (_ for _ in ()).throw(
            AssertionError(
                "LLMInvoke.invoke was called during structure_extract ingest"
            )
        ),
    )

    result = ingest(
        ws, src, source_id="src.java.project", source_class="code"
    )
    assert result.succeeded
    assert result.policy == "structure_extract"
    assert result.structure_reports == [
        "state/structure/src.java.project.yaml"
    ]
    assert result.claims_created == []
    assert result.pages_rendered == []

    loaded = yaml_mod.safe_load(
        ws.structure_report_path("src.java.project").read_text(encoding="utf-8")
    )
    assert loaded["version"] == "structure_report_v1"
    assert loaded["language"] == "java"
    assert loaded["source_class"] == "code"
    paths = {(it["symbol_path"], it["kind"]) for it in loaded["items"]}
    assert paths == {
        ("Project", "class"),
        ("Project.Project", "constructor"),
        ("Project.evaluate", "method"),
        ("Project.maxsize", "field"),
        ("Metric", "interface"),
        ("PatchKind", "enum"),
        ("PatchRecord", "record"),
    }
