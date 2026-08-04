"""Contract tests for the llloom MCP server surface.

The MCP SDK is an optional dependency. These tests exercise the
package-owned tool handlers directly (which do not need the SDK) and
prove the ``llloom-mcp`` entry point refuses cleanly when the
optional extra is absent.
"""

from __future__ import annotations

import importlib
import io
import sys
import types
from contextlib import redirect_stderr
from pathlib import Path

import pytest

from llloom.claims.models import (
    Assertion,
    EntityContainer,
    Evidence,
    Locator,
    Relation,
)
from llloom.claims.store import ClaimStore
from llloom.llm.harness import LLMInvoke
from llloom.mcp_server import tools as tool_module
from llloom.mcp_server.tools import (
    TOOL_NAMES,
    MCPToolError,
    to_jsonable,
    tool_graph_neighbors,
    tool_lint,
    tool_list_merge_proposals,
    tool_query,
    tool_status,
    tool_verify,
)
from llloom.ops.rebuild import rebuild
from llloom.workspace.layout import Workspace


SOURCE = """\
# Article

## Methods

Complementarity prioritizes sites that add features not already represented in the selected set.
"""


def _seed_two_entity_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    store = ClaimStore(ws)
    alpha = EntityContainer(
        entity_id="concept.alpha",
        entity_type="concept",
        display_name="Alpha",
        assertions=[
            Assertion(
                claim_id="c.alpha.1",
                subject_id="concept.alpha",
                claim_kind="definition",
                claim_text="alpha",
                evidence=[
                    Evidence(
                        source_id="src.stub",
                        locator=Locator(
                            locator_type="markdown_prose_v1",
                            heading_path=["Methods"],
                            paragraph_index=1,
                            sentence_start=1,
                            sentence_end=1,
                        ),
                        excerpt_hash="0" * 64,
                    )
                ],
            )
        ],
        relations=[
            Relation(
                relation_id="r.alpha_supports_beta",
                source_claim_id="c.alpha.1",
                relation_type="supports",
                target_claim_id="c.beta.1",
                status="active",
            )
        ],
    )
    beta = EntityContainer(
        entity_id="concept.beta",
        entity_type="concept",
        display_name="Beta",
        assertions=[
            Assertion(
                claim_id="c.beta.1",
                subject_id="concept.beta",
                claim_kind="definition",
                claim_text="beta",
                evidence=[
                    Evidence(
                        source_id="src.stub",
                        locator=Locator(
                            locator_type="markdown_prose_v1",
                            heading_path=["Methods"],
                            paragraph_index=1,
                            sentence_start=1,
                            sentence_end=1,
                        ),
                        excerpt_hash="0" * 64,
                    )
                ],
            )
        ],
    )
    store.save_entity(alpha)
    store.save_entity(beta)
    return ws


# --- 1. import isolation: base package does not require mcp -------------


def test_importing_llloom_does_not_require_mcp_sdk() -> None:
    """``llloom``, ``llloom.llm``, ``llloom.ops``, ``llloom.state``,
    and ``llloom.mcp_server.tools`` must all be importable without the
    MCP SDK. ``llloom.mcp_server.server`` must also be importable
    because its SDK import is lazy; only running ``main()`` requires
    the extra."""
    for name in (
        "llloom",
        "llloom.llm",
        "llloom.ops",
        "llloom.state",
        "llloom.mcp_server",
        "llloom.mcp_server.tools",
        "llloom.mcp_server.server",
    ):
        mod = importlib.import_module(name)
        assert mod is not None
    # The server module must not eagerly bind a real `mcp` module.
    server_mod = sys.modules["llloom.mcp_server.server"]
    assert "mcp" not in server_mod.__dict__ or not isinstance(
        server_mod.__dict__.get("mcp"), types.ModuleType
    )


def test_mcp_server_main_refuses_cleanly_when_sdk_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a machine without ``mcp`` installed. ``llloom-mcp``
    must exit non-zero with a helpful message naming ``llloom[mcp]``
    and must not raise a raw ImportError traceback."""
    ws = Workspace.init(tmp_path)

    class _Blocker:
        def find_spec(self, name, path=None, target=None):  # noqa: ARG002
            if name == "mcp" or name.startswith("mcp."):
                raise ImportError("no mcp SDK")
            return None

    for mod_name in list(sys.modules):
        if mod_name == "mcp" or mod_name.startswith("mcp."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])

    from llloom.mcp_server.server import main as mcp_main

    err = io.StringIO()
    with redirect_stderr(err):
        rc = mcp_main(["--root", str(ws.root)])
    assert rc == 2
    assert "llloom[mcp]" in err.getvalue()


# --- 2. declared tool names match expectations -------------------------


def test_tool_names_are_read_only_and_diagnostic() -> None:
    expected = {
        "llloom_status",
        "llloom_query",
        "llloom_verify",
        "llloom_lint",
        "llloom_graph_neighbors",
        "llloom_list_merge_proposals",
    }
    assert set(TOOL_NAMES) == expected
    forbidden = {
        "llloom_ingest",
        "llloom_retract",
        "llloom_promote",
        "llloom_merge_alias",
        "llloom_review_alias",
        "llloom_reject_alias",
        "llloom_unlock",
        "llloom_reconcile",
        "llloom_rebuild",
    }
    assert not forbidden.intersection(TOOL_NAMES)


# --- 3. tool_status returns a JSON-compatible dict ----------------------


def test_tool_status_returns_jsonable_dict(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    out = tool_status(ws)
    assert isinstance(out, dict)
    # JSON primitives only. No dataclass instances leaked through.
    import json

    json.dumps(out)
    assert "source_count" in out
    assert "claim_count" in out


# --- 4. tool_query never invokes LLMInvoke ------------------------------


def test_tool_query_does_not_invoke_llm_invoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace.init(tmp_path)

    def _fail(self, **kwargs):  # noqa: ANN001 - test stub
        raise AssertionError(
            "MCP tool_query must not invoke LLMInvoke; "
            f"got operation_kind={kwargs.get('operation_kind')!r}"
        )

    monkeypatch.setattr(LLMInvoke, "invoke", _fail)

    out = tool_query(ws, question="alpha")
    assert isinstance(out, dict)
    assert out["question"] == "alpha"
    # Empty workspace produces the canonical empty-shape QueryResult.
    assert isinstance(out["citations"], list)
    assert isinstance(out["used_claim_ids"], list)
    assert isinstance(out["used_verbatim_spans"], list)


def test_tool_query_refuses_empty_question(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    with pytest.raises(MCPToolError):
        tool_query(ws, question="")


# --- 5. tool_verify returns VerifyResult shape --------------------------


def test_tool_verify_returns_verify_result_shape(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    out = tool_verify(ws)
    assert isinstance(out, dict)
    for key in ("verified", "failed", "notes", "mismatches", "passed"):
        assert key in out
    assert isinstance(out["mismatches"], list)


# --- 6. tool_lint preserves generated_canary plumbing -------------------


def test_tool_lint_default_and_generated_canary(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    default = tool_lint(ws)
    assert isinstance(default, dict)
    assert "canary_hits" in default
    with_canary = tool_lint(ws, generated_canary=True)
    assert isinstance(with_canary, dict)
    assert "canary_hits" in with_canary


# --- 7. tool_graph_neighbors: rehydrated edge + limit=0 + bad direction --


def test_tool_graph_neighbors_returns_rehydrated_edges(tmp_path: Path) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    rebuild(ws, target="graph")

    edges = tool_graph_neighbors(
        ws, claim_id="c.alpha.1", direction="out"
    )
    assert isinstance(edges, list)
    assert len(edges) == 1
    edge = edges[0]
    assert edge["relation_id"] == "r.alpha_supports_beta"
    assert edge["source_entity_id"] == "concept.alpha"
    assert edge["target_entity_id"] == "concept.beta"


def test_tool_graph_neighbors_respects_limit_zero(tmp_path: Path) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    rebuild(ws, target="graph")
    assert tool_graph_neighbors(
        ws, claim_id="c.alpha.1", direction="out", limit=0
    ) == []
    assert tool_graph_neighbors(
        ws, claim_id="c.alpha.1", direction="out", limit=-5
    ) == []


def test_tool_graph_neighbors_unknown_direction_raises_tool_error(
    tmp_path: Path,
) -> None:
    ws = _seed_two_entity_workspace(tmp_path)
    with pytest.raises(MCPToolError) as excinfo:
        tool_graph_neighbors(ws, claim_id="c.alpha.1", direction="sideways")
    assert "sideways" in str(excinfo.value)


# --- 8. tool_list_merge_proposals returns JSON-compatible list ----------


def test_tool_list_merge_proposals_returns_list(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    out = tool_list_merge_proposals(ws)
    assert isinstance(out, list)


# --- 9. to_jsonable handles dataclasses, Path, nested structures --------


def test_to_jsonable_walks_tree_and_preserves_shape(tmp_path: Path) -> None:
    import json

    from llloom.ops.results import StatusResult, VerbatimSpan

    dc = StatusResult(
        source_count=1,
        claim_count=2,
        rendered_page_count=0,
        pending_review_count=0,
        stale_count=0,
        retracted_count=0,
        lock_held=False,
        lock_owner=None,
        last_operation_id=None,
        last_operation_status=None,
    )
    out = to_jsonable(dc)
    assert isinstance(out, dict)
    assert out["source_count"] == 1
    assert out["lock_owner"] is None
    json.dumps(out)

    # Mixed nesting: list of dataclasses, Path, dict.
    spans = [VerbatimSpan("s", "exc", "h", 0, 3), VerbatimSpan("t", "e2", "h2", 1, 2)]
    walked = to_jsonable(
        {
            "workspace": tmp_path,
            "spans": spans,
            "flag": True,
            "empty": None,
            "tuple": (1, 2, 3),
        }
    )
    assert walked["workspace"] == str(tmp_path)
    assert walked["tuple"] == [1, 2, 3]
    assert walked["spans"][0]["source_id"] == "s"
    assert walked["flag"] is True
    assert walked["empty"] is None
    json.dumps(walked)


# --- 10. load_bound_workspace: single-root binding ----------------------


def test_load_bound_workspace_rejects_invalid_root(tmp_path: Path) -> None:
    from llloom.mcp_server.server import load_bound_workspace

    # Path exists but is not an initialized workspace.
    bogus = tmp_path / "not_a_workspace"
    bogus.mkdir()
    with pytest.raises(Exception):
        load_bound_workspace(bogus)


def test_main_refuses_invalid_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from llloom.mcp_server.server import main as mcp_main

    bogus = tmp_path / "does_not_exist"
    rc = mcp_main(["--root", str(bogus)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "invalid workspace" in captured.err


# --- 11. tool handlers do not accept an alternate root ------------------


def test_tool_handlers_do_not_accept_root_kwarg() -> None:
    """Every public tool handler must take a ``Workspace`` positionally
    and never a ``root``/``path``/``workspace_root`` string keyword.
    This is the structural proof that tool calls cannot redirect to a
    different workspace once the server has bound one."""
    import inspect

    for name in (
        "tool_status",
        "tool_query",
        "tool_verify",
        "tool_lint",
        "tool_graph_neighbors",
        "tool_list_merge_proposals",
    ):
        sig = inspect.signature(getattr(tool_module, name))
        params = sig.parameters
        forbidden = {"root", "workspace_root", "path", "ws_root"}
        assert not forbidden.intersection(params), (
            f"{name} accepts a forbidden root-override kwarg: {params}"
        )
        # First positional parameter is always the workspace.
        first = next(iter(params))
        assert first == "workspace", (
            f"{name} first parameter should be 'workspace', got {first!r}"
        )
