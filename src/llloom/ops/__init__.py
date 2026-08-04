"""User-facing operations.

Each operation returns a structured result object. The CLI and any future
MCP wrapper consume those results; callers should not need to parse
strings to drive behavior.
"""

from llloom.ops.results import (
    AcceptedDoctorWarning,
    ClaimCard,
    ClaimSummary,
    CreatedClaim,
    DoctorResult,
    DoctorWarning,
    EvidenceSummary,
    HealthReport,
    IngestResult,
    PageCreateResult,
    PageSummary,
    PlannedSeedClaim,
    SeedManifestResult,
    LintResult,
    MergeProposalSummary,
    PdfPrepArtifact,
    PdfPrepResult,
    PromoteResult,
    QueryResult,
    ReconcileResult,
    RenderResult as OpsRenderResult,
    RenderTargetListEntry,
    RenderTargetSummary,
    RetractResult,
    SourceSummary,
    StatusResult,
    SupersedeResult,
    UnlockRecord,
    UpdateReviewBundle,
    VerifyResult,
)
from llloom.ops.ingest import ingest
from llloom.ops.verify import verify
from llloom.ops.render import render
from llloom.ops.lint import lint
from llloom.ops.reconcile import reconcile
from llloom.ops.unlock import unlock
from llloom.ops.promote import promote
from llloom.ops.retract import retract
from llloom.ops.rebuild import rebuild
from llloom.ops.query import query
from llloom.ops.prepare_pdf import prepare_pdf
from llloom.ops.alias import (
    list_merge_proposals,
    review_alias,
    merge_alias,
    reject_alias,
)
from llloom.ops.status import status
from llloom.ops.seed import apply_seed_manifest
from llloom.ops.page import PageCreateError, create_page
from llloom.ops.supersede import SupersedeError, supersede
from llloom.ops.doctor import doctor
from llloom.ops.inspect import (
    ClaimCardError,
    InspectFilterError,
    claim_card,
    list_claims,
    list_pages,
    list_render_targets,
    list_sources,
)

__all__ = [
    "AcceptedDoctorWarning",
    "ClaimCard",
    "ClaimCardError",
    "ClaimSummary",
    "CreatedClaim",
    "DoctorResult",
    "DoctorWarning",
    "EvidenceSummary",
    "HealthReport",
    "IngestResult",
    "InspectFilterError",
    "LintResult",
    "MergeProposalSummary",
    "OpsRenderResult",
    "PageCreateError",
    "PageCreateResult",
    "PageSummary",
    "PdfPrepArtifact",
    "PdfPrepResult",
    "PlannedSeedClaim",
    "PromoteResult",
    "QueryResult",
    "ReconcileResult",
    "RenderTargetListEntry",
    "RenderTargetSummary",
    "RetractResult",
    "SeedManifestResult",
    "SourceSummary",
    "StatusResult",
    "SupersedeError",
    "SupersedeResult",
    "UnlockRecord",
    "UpdateReviewBundle",
    "VerifyResult",
    "apply_seed_manifest",
    "claim_card",
    "create_page",
    "doctor",
    "ingest",
    "verify",
    "render",
    "lint",
    "reconcile",
    "unlock",
    "promote",
    "retract",
    "rebuild",
    "query",
    "prepare_pdf",
    "list_claims",
    "list_merge_proposals",
    "list_pages",
    "list_render_targets",
    "list_sources",
    "review_alias",
    "merge_alias",
    "reject_alias",
    "status",
    "supersede",
]

