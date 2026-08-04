"""llloom - source-grounded knowledge compiler with wiki views.

Phase 2 first vertical slice. Implements the authority core from
04_specification/: workspace loading, typed ingest, per-claim span
verification, variant-(B) page rendering, typed-input LLMInvoke harness,
workspace-scoped file lock with journal-backed reconcile, lifecycle and
alias queue verbs, read-only authoritative query, canary-enforced
exclusion contract.
"""

from llloom.workspace import Workspace

__all__ = ["Workspace"]
__version__ = "0.1.0"

