"""Local MCP server surface for llloom.

This first slice exposes **read-only and diagnostic** tools only:
``llloom_status``, ``llloom_query``, ``llloom_verify``,
``llloom_lint``, ``llloom_graph_neighbors``, and
``llloom_list_merge_proposals``. Mutating operations (ingest,
retract, promote, alias merge/reject, unlock, reconcile, rebuild)
are deliberately withheld until the server plumbing + tool-shape
contract is proven on safe surfaces.

The MCP SDK is an optional dependency. Install with
``pip install "llloom[mcp]"``. Importing this subpackage does
**not** require the SDK; only running :func:`server.main` does.
"""

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

__all__ = [
    "MCPToolError",
    "TOOL_NAMES",
    "to_jsonable",
    "tool_graph_neighbors",
    "tool_lint",
    "tool_list_merge_proposals",
    "tool_query",
    "tool_status",
    "tool_verify",
]
