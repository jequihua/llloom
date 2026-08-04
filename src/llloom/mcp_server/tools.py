"""Pure, SDK-free tool handlers for the llloom MCP server.

Each handler takes an already-loaded :class:`Workspace` plus the
tool's keyword inputs and returns a JSON-compatible dict or list.
Handlers wrap existing ``llloom`` library operations — they do not
own new state, do not bypass ``Workspace.load``, and do not invoke
``LLMInvoke``.

Importing this module does **not** require the MCP SDK; the SDK is
only imported by :mod:`llloom.mcp_server.server` when the user runs
``llloom-mcp``.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from llloom.ops.alias import list_merge_proposals
from llloom.ops.lint import lint
from llloom.ops.query import query
from llloom.ops.status import status
from llloom.ops.verify import verify
from llloom.state.graph import GraphSidecarError, graph_neighbors
from llloom.workspace.layout import Workspace


TOOL_NAMES: tuple[str, ...] = (
    "llloom_status",
    "llloom_query",
    "llloom_verify",
    "llloom_lint",
    "llloom_graph_neighbors",
    "llloom_list_merge_proposals",
)


class MCPToolError(Exception):
    """Raised when a tool handler refuses a call.

    The MCP server translates this into the SDK's error shape; the
    message is safe to return to the caller.
    """


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, lists, tuples, dicts, and
    :class:`Path` instances into JSON-compatible primitives.

    Existing result dataclasses (``StatusResult``, ``QueryResult``,
    ``VerifyResult``, ``LintResult``, ``VerbatimSpan``,
    ``VerifierMismatch``, ``GraphEdge``, ``MergeProposalSummary``) are
    preserved shape-for-shape — the helper only walks the tree.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def tool_status(workspace: Workspace) -> dict:
    return to_jsonable(status(workspace))


def tool_query(workspace: Workspace, *, question: str) -> dict:
    if not isinstance(question, str) or not question:
        raise MCPToolError("llloom_query requires a non-empty 'question' string")
    return to_jsonable(query(workspace, question=question))


def tool_verify(workspace: Workspace, *, target: str | None = None) -> dict:
    return to_jsonable(verify(workspace, target=target))


def tool_lint(workspace: Workspace, *, generated_canary: bool = False) -> dict:
    return to_jsonable(lint(workspace, generated_canary=bool(generated_canary)))


def tool_graph_neighbors(
    workspace: Workspace,
    *,
    claim_id: str,
    direction: str = "both",
    relation_types: list[str] | None = None,
    include_inactive: bool = False,
    limit: int = 50,
) -> list[dict]:
    if not isinstance(claim_id, str) or not claim_id:
        raise MCPToolError(
            "llloom_graph_neighbors requires a non-empty 'claim_id' string"
        )
    types_set = set(relation_types) if relation_types else None
    try:
        edges = graph_neighbors(
            workspace,
            claim_id=claim_id,
            direction=direction,
            relation_types=types_set,
            include_inactive=bool(include_inactive),
            limit=int(limit),
        )
    except GraphSidecarError as exc:
        raise MCPToolError(str(exc)) from exc
    return to_jsonable(edges)


def tool_list_merge_proposals(workspace: Workspace) -> list[dict]:
    return to_jsonable(list_merge_proposals(workspace))
