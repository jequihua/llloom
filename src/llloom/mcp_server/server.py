"""Local stdio MCP server for llloom (read-only + diagnostics).

Entry point installed as the ``llloom-mcp`` console script. The MCP
SDK (``mcp``) is an optional dependency; this module imports it
lazily inside :func:`main` so ``import llloom`` and its submodules
never require ``pip install "llloom[mcp]"``.

First slice surface: stdio transport, one workspace bound at
startup, and the six tool handlers defined in
:mod:`llloom.mcp_server.tools`. No HTTP / SSE / WebSocket
transport, no background mode, no mutation tools.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llloom.mcp_server.tools import (
    TOOL_NAMES,
    MCPToolError,
    tool_graph_neighbors,
    tool_lint,
    tool_list_merge_proposals,
    tool_query,
    tool_status,
    tool_verify,
)
from llloom.workspace.layout import Workspace


class MCPServerError(Exception):
    """Raised for MCP-server-level startup failures (missing SDK,
    invalid workspace). Caller-facing message; never includes
    secrets."""


def load_bound_workspace(root: Path | str) -> Workspace:
    """Load and validate the single workspace this server instance
    will serve. One workspace per process; tool calls do not accept
    an alternate root."""
    return Workspace.load(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llloom-mcp",
        description=(
            "Local stdio MCP server exposing read-only and diagnostic "
            "llloom operations. Install with: pip install \"llloom[mcp]\"."
        ),
    )
    parser.add_argument(
        "--root",
        default=".",
        help="workspace root to bind (default: current working directory)",
    )
    parser.add_argument(
        "--name",
        default="llloom",
        help="MCP server name advertised to clients (default: llloom)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        workspace = load_bound_workspace(args.root)
    except Exception as exc:
        sys.stderr.write(f"llloom-mcp: invalid workspace at {args.root!r}: {exc}\n")
        return 2

    try:
        server = _build_mcp_server(workspace, name=args.name)
    except MCPServerError as exc:
        sys.stderr.write(f"llloom-mcp: {exc}\n")
        return 2

    try:
        _run_stdio(server)
    except Exception as exc:  # noqa: BLE001 - surface cleanly, no traceback
        sys.stderr.write(f"llloom-mcp: server terminated: {exc}\n")
        return 1
    return 0


def _build_mcp_server(workspace: Workspace, *, name: str):
    """Construct the MCP SDK server object and register tools.

    Imported lazily so a missing MCP SDK fails only at
    ``llloom-mcp`` invocation, not at ``import llloom``.
    """
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MCPServerError(
            "the MCP optional dependency is not installed; "
            "install it with: pip install \"llloom[mcp]\""
        ) from exc

    server = FastMCP(name)

    @server.tool(name="llloom_status")
    def _status() -> dict:
        return tool_status(workspace)

    @server.tool(name="llloom_query")
    def _query(question: str) -> dict:
        try:
            return tool_query(workspace, question=question)
        except MCPToolError as exc:
            raise ValueError(str(exc)) from exc

    @server.tool(name="llloom_verify")
    def _verify(target: str | None = None) -> dict:
        return tool_verify(workspace, target=target)

    @server.tool(name="llloom_lint")
    def _lint(generated_canary: bool = False) -> dict:
        return tool_lint(workspace, generated_canary=generated_canary)

    @server.tool(name="llloom_graph_neighbors")
    def _graph_neighbors(
        claim_id: str,
        direction: str = "both",
        relation_types: list[str] | None = None,
        include_inactive: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        try:
            return tool_graph_neighbors(
                workspace,
                claim_id=claim_id,
                direction=direction,
                relation_types=relation_types,
                include_inactive=include_inactive,
                limit=limit,
            )
        except MCPToolError as exc:
            raise ValueError(str(exc)) from exc

    @server.tool(name="llloom_list_merge_proposals")
    def _list_merge_proposals() -> list[dict]:
        return tool_list_merge_proposals(workspace)

    _assert_tool_set(server)
    return server


def _assert_tool_set(server) -> None:
    """Defence-in-depth: refuse to start if any unexpected tool
    slipped into the set, or if any required read-only tool is
    missing. The exact attribute used to inspect registered tools
    is SDK-version-dependent, so this is best-effort and silent
    when the SDK does not expose an introspection surface."""
    _forbidden = {
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
    registered = _registered_tool_names(server)
    if registered is None:
        return
    leaked = sorted(name for name in registered if name in _forbidden)
    if leaked:
        raise MCPServerError(
            f"refusing to start: mutating tool(s) registered: {leaked}"
        )


def _registered_tool_names(server) -> set[str] | None:
    """Best-effort introspection of the FastMCP tool registry.

    Returns ``None`` when the SDK does not expose a tool-name
    collection this helper knows about; the tool-set assertion
    above then becomes a no-op and the public `TOOL_NAMES`
    constant remains the declared contract."""
    for attr in ("_tools", "tools"):
        maybe = getattr(server, attr, None)
        if isinstance(maybe, dict):
            return {str(k) for k in maybe.keys()}
        if isinstance(maybe, (list, tuple, set)):
            return {
                str(getattr(t, "name", t)) for t in maybe
            }
    return None


def _run_stdio(server) -> None:
    """Run the stdio transport. The SDK handles the event loop."""
    server.run()
