"""Claim lifecycle state machine (Slice 078).

Centralizes the legal lifecycle transition graph that lives in the
roadmap and the ``04_specification`` docs so every caller — promote,
supersede, future doctor/review surfaces — consults a single source of
truth instead of carrying its own private transition table.

The graph:

```text
draft -> reviewed -> validated -> superseded
                              \\-> archived
```

Source-cascade statuses (``retracted``, ``retracted_by_source``,
``stale``) and the terminal lifecycle states (``superseded``,
``archived``, ``retracted``, ``retracted_by_source``, ``stale``) are
not normal operator-driven promotion targets — they remain valid
``CLAIM_STATUSES`` because the source-retraction cascade still moves
claims into them — but
:data:`LEGAL_LIFECYCLE_TRANSITIONS` deliberately omits any edge
that would re-promote them silently.
"""

from __future__ import annotations

from llloom.claims.models import CLAIM_STATUSES


# Canonical promotion graph. Every entry is a ``(from_status,
# to_status)`` tuple. The set is frozen so callers can use it as a
# dict key, set member, or test assertion target without copying.
LEGAL_LIFECYCLE_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("draft", "reviewed"),
        ("reviewed", "validated"),
        ("validated", "superseded"),
        ("validated", "archived"),
    }
)


# Recommended promotion path for every non-terminal lifecycle state.
# Used by :func:`explain_transition_refusal` to give operators a
# concrete, actionable next-step suggestion. The chain is rooted at
# ``draft`` and ends at the two normal terminals ``superseded`` and
# ``archived``.
_NORMAL_PROMOTION_PATH: tuple[str, ...] = (
    "draft",
    "reviewed",
    "validated",
)


# Source-cascade statuses. These are reachable only through the
# ``retract`` cascade — operator-driven promotion / supersede paths
# refuse to touch them so a source retraction cannot be silently
# undone by lifecycle plumbing.
SOURCE_CASCADE_STATUSES: frozenset[str] = frozenset(
    {"retracted", "retracted_by_source", "stale"}
)


def can_transition(from_status: str, to_status: str) -> bool:
    """Return ``True`` iff ``(from_status, to_status)`` is a legal
    operator-driven lifecycle transition.

    Validates both arguments against :data:`CLAIM_STATUSES`. Returns
    ``False`` for any pair the legal graph does not contain — that
    includes self-edges (``draft -> draft``), reverse edges
    (``validated -> reviewed``), and any edge originating from a
    source-cascade or terminal state.
    """
    if from_status not in CLAIM_STATUSES or to_status not in CLAIM_STATUSES:
        return False
    return (from_status, to_status) in LEGAL_LIFECYCLE_TRANSITIONS


def explain_transition_refusal(from_status: str, to_status: str) -> str:
    """Build an actionable human-facing refusal message for an
    illegal lifecycle transition.

    The message names the current status, the requested status, and
    the concrete promotion path the caller should take instead (when
    the requested status is reachable through the normal chain).
    """
    if from_status not in CLAIM_STATUSES:
        return (
            f"current status {from_status!r} is not a known lifecycle state; "
            f"allowed states: {sorted(CLAIM_STATUSES)}"
        )
    if to_status not in CLAIM_STATUSES:
        return (
            f"requested status {to_status!r} is not a known lifecycle state; "
            f"allowed states: {sorted(CLAIM_STATUSES)}"
        )
    if from_status in SOURCE_CASCADE_STATUSES:
        return (
            f"claim is {from_status!r}; source-cascade statuses cannot be "
            "promoted operator-side. Re-ingest the source or run an "
            "explicit lifecycle rebuild instead."
        )
    if from_status == to_status:
        return (
            f"claim is already {to_status!r}; no-op transitions are not "
            "allowed"
        )
    suggested = _suggest_path(from_status, to_status)
    if suggested is not None:
        return (
            f"transition {from_status!r} -> {to_status!r} not allowed; "
            f"promote through {suggested} first"
        )
    return (
        f"transition {from_status!r} -> {to_status!r} not allowed; "
        f"legal transitions: {sorted(LEGAL_LIFECYCLE_TRANSITIONS)}"
    )


def _suggest_path(from_status: str, to_status: str) -> str | None:
    """Return a human-readable suggested promotion chain when
    ``to_status`` sits downstream of ``from_status`` on the normal
    promotion path, otherwise ``None``.
    """
    chain = list(_NORMAL_PROMOTION_PATH) + ["superseded", "archived"]
    if from_status not in chain or to_status not in chain:
        return None
    from_idx = chain.index(from_status)
    to_idx = chain.index(to_status)
    # Only suggest when ``to_status`` is strictly downstream of
    # ``from_status`` on the chain.
    if to_idx <= from_idx:
        return None
    # Build the intermediate-states suggestion ("reviewed then
    # validated"). For supersession the path is the same up through
    # validated; the final hop is added implicitly.
    intermediates: list[str] = []
    for idx in range(from_idx + 1, min(to_idx, len(_NORMAL_PROMOTION_PATH))):
        intermediates.append(_NORMAL_PROMOTION_PATH[idx])
    if not intermediates:
        return None
    if len(intermediates) == 1:
        return repr(intermediates[0])
    return " then ".join(repr(s) for s in intermediates)
