"""`doctor` operation (Slice 079): read-only workspace diagnostics.

`llloom doctor` packages the field-runbook health checks into one
mechanical, read-only surface. It detects the same drift signals
the existing `rebuild health_report` walk surfaces (stale lock,
interrupted journal, stale render fingerprint, missing sidecars,
missing / stale structure reports, lifecycle / source / page
anomalies) and adds two new capabilities:

- abandoned render-transaction directory detection
  (`state/transactions/<op_id>/`);
- an **accepted-warnings allowlist** so reviewers can explicitly
  retire known signals with evidence, separating them from new
  warnings without removing them from the report.

The operation is **strictly read-only**: no `operation(...)`
context, no workspace lock acquisition, no journal entry, no
sidecar / page / fingerprint / claim / source / report write, no
model / provider call. The doctor is the read-only complement to
the existing `reconcile` / per-target `rebuild` repair surfaces.

`doctor(workspace, op_id=...)` and `doctor(workspace,
last_op=True)` additionally build a :class:`UpdateReviewBundle`
for the selected operation: journal fields, the Slice 076 seed
update report (when one exists at
`state/reports/updates/<op_id>.yaml`), and current lint / verify
/ status summaries.

See `04_specification/operations_and_cli.md` `## doctor` for the
full contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from llloom.claims.lifecycle import SOURCE_CASCADE_STATUSES
from llloom.claims.store import ClaimStore
from llloom.ops.lint import lint as run_lint
from llloom.ops.results import (
    AcceptedDoctorWarning,
    DoctorResult,
    DoctorWarning,
    UpdateReviewBundle,
)
from llloom.ops.status import status as run_status
from llloom.ops.verify import verify as run_verify
from llloom.pages.regions import PageParseError, parse_page
from llloom.schema.policy import SchemaError, load_schema
from llloom.sources.registry import SourceRegistry
from llloom.state.fingerprints import FingerprintStore
from llloom.state.journal import OperationJournal
from llloom.state.lock import (
    LockError,
    PID_STATE_ALIVE,
    PID_STATE_DEAD,
    PID_STATE_UNKNOWN,
    WorkspaceLock,
    local_owner_pid_state,
)
from llloom.pages.render import compute_page_render_fingerprints
from llloom.workspace.layout import Workspace


# Optional default path for an accepted-warnings allowlist. The
# directory is the same `state/reports/health/` Slice 053 already
# uses for derived health artifacts; the file is optional and the
# doctor never creates it.
_DEFAULT_ACCEPTED_PATH = "state/reports/health/accepted_warnings.yaml"
_ACCEPTED_VERSION = "accepted_warnings_v1"


def doctor(
    workspace: Workspace,
    *,
    op_id: str | None = None,
    last_op: bool = False,
    accepted_warnings: str | Path | None = None,
) -> DoctorResult:
    """Read-only workspace diagnostic surface.

    The operation walks the registry, claim store, lock + journal,
    fingerprint store, page tree, and sidecars to surface a list of
    :class:`DoctorWarning` records with stable ids and recommended
    next commands. It never writes anything: no `operation(...)`
    context is opened, no lock is acquired, no journal entry is
    created, no canonical state is mutated.

    ``accepted_warnings``: optional path to a YAML allowlist
    (`accepted_warnings_v1`). When absent, the doctor looks for
    `state/reports/health/accepted_warnings.yaml`; when present and
    well-formed, every current warning whose `warning_id` matches an
    allowlist entry moves from `warnings` to `accepted_warnings`.
    Allowlist entries that match no current warning surface in
    `stale_acceptances`.

    ``op_id`` / ``last_op``: at most one may be set. When set, the
    result's ``update_review`` field is populated with an
    :class:`UpdateReviewBundle` for the selected operation.
    """
    if op_id is not None and last_op:
        raise ValueError(
            "doctor: pass at most one of op_id / last_op"
        )

    # ---- accepted-warnings allowlist --------------------------------
    accepted_entries, accepted_load_warning = _load_accepted_warnings(
        workspace, accepted_warnings
    )

    # ---- detector chain ---------------------------------------------
    warnings: list[DoctorWarning] = []
    warnings.extend(_detect_lock_and_journal(workspace))
    warnings.extend(_detect_transactions(workspace))
    warnings.extend(_detect_render_drift(workspace))
    warnings.extend(_detect_sidecars(workspace))
    warnings.extend(_detect_structure_reports(workspace))
    warnings.extend(_detect_lifecycle(workspace))
    warnings.extend(_detect_sources(workspace))
    warnings.extend(_detect_pages(workspace))

    if accepted_load_warning is not None:
        warnings.append(accepted_load_warning)

    # Deterministic sort: by (severity rank, category, warning_id).
    warnings.sort(key=_warning_sort_key)

    # ---- accepted-warning separation --------------------------------
    accepted_by_id = {entry["warning_id"]: entry for entry in accepted_entries}
    remaining: list[DoctorWarning] = []
    accepted: list[AcceptedDoctorWarning] = []
    matched_accepted_ids: set[str] = set()
    for w in warnings:
        if w.warning_id in accepted_by_id:
            entry = accepted_by_id[w.warning_id]
            accepted.append(
                AcceptedDoctorWarning(
                    warning=w,
                    accepted_reason=entry.get("reason", ""),
                    accepted_by=entry.get("accepted_by"),
                    accepted_at=entry.get("accepted_at"),
                    evidence_links=list(entry.get("evidence", []) or []),
                )
            )
            matched_accepted_ids.add(w.warning_id)
        else:
            remaining.append(w)

    stale_acceptances = sorted(
        wid for wid in accepted_by_id if wid not in matched_accepted_ids
    )

    recommended = _aggregate_recommendations(remaining)

    # ---- update review bundle ---------------------------------------
    bundle: UpdateReviewBundle | None = None
    if op_id is not None or last_op:
        bundle = _build_update_review_bundle(
            workspace,
            op_id=op_id,
            last_op=last_op,
            warnings=remaining,
            accepted=accepted,
        )

    return DoctorResult(
        target="doctor",
        warning_count=len(remaining),
        accepted_warning_count=len(accepted),
        warnings=remaining,
        accepted_warnings=accepted,
        stale_acceptances=stale_acceptances,
        recommended_next_commands=recommended,
        update_review=bundle,
    )


# ---- accepted-warnings ------------------------------------------------


def _load_accepted_warnings(
    workspace: Workspace, override: str | Path | None
) -> tuple[list[dict[str, Any]], DoctorWarning | None]:
    """Return ``(entries, malformed_warning)``. ``entries`` is a list
    of validated allowlist dicts (each with a non-empty ``warning_id``,
    ``reason``, and ``evidence`` list). ``malformed_warning`` is set
    when the file exists but does not match the
    ``accepted_warnings_v1`` shape — the doctor surfaces a single
    `accepted-warnings:malformed-file` warning rather than crashing.
    """
    if override is not None:
        path = Path(override)
        if not path.is_absolute():
            path = workspace.root / path
    else:
        path = workspace.root / _DEFAULT_ACCEPTED_PATH

    if not path.is_file():
        return [], None

    rel = _rel(workspace, path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [], DoctorWarning(
            warning_id=f"accepted-warnings:malformed-file:{rel}",
            severity="warning",
            category="accepted-warnings",
            message=(
                f"accepted-warnings file {rel!r} failed to parse: {exc}; "
                "no allowlist entries applied"
            ),
            recommended_command=None,
            evidence=[rel],
        )

    if not isinstance(raw, dict) or raw.get("version") != _ACCEPTED_VERSION:
        return [], DoctorWarning(
            warning_id=f"accepted-warnings:malformed-file:{rel}",
            severity="warning",
            category="accepted-warnings",
            message=(
                f"accepted-warnings file {rel!r} is missing the "
                f"{_ACCEPTED_VERSION!r} version tag; no allowlist entries "
                "applied"
            ),
            recommended_command=None,
            evidence=[rel],
        )

    accepted_raw = raw.get("accepted") or []
    if not isinstance(accepted_raw, list):
        return [], DoctorWarning(
            warning_id=f"accepted-warnings:malformed-file:{rel}",
            severity="warning",
            category="accepted-warnings",
            message=(
                f"accepted-warnings file {rel!r} 'accepted' field must be "
                "a list; no allowlist entries applied"
            ),
            recommended_command=None,
            evidence=[rel],
        )

    entries: list[dict[str, Any]] = []
    malformed: list[str] = []
    for index, entry in enumerate(accepted_raw):
        if not isinstance(entry, dict):
            malformed.append(f"[{index}]: not a mapping")
            continue
        wid = entry.get("warning_id")
        reason = entry.get("reason")
        evidence = entry.get("evidence")
        if not isinstance(wid, str) or not wid:
            malformed.append(f"[{index}]: missing or empty 'warning_id'")
            continue
        if not isinstance(reason, str) or not reason.strip():
            malformed.append(
                f"[{index}]: 'reason' must be a non-empty string for "
                f"warning_id={wid!r}"
            )
            continue
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(e, str) and e for e in evidence)
        ):
            malformed.append(
                f"[{index}]: 'evidence' must list at least one non-empty "
                f"string for warning_id={wid!r}"
            )
            continue
        entries.append(
            {
                "warning_id": wid,
                "reason": reason,
                "accepted_by": entry.get("accepted_by"),
                "accepted_at": entry.get("accepted_at"),
                "evidence": evidence,
            }
        )

    malformed_warning: DoctorWarning | None = None
    if malformed:
        malformed_warning = DoctorWarning(
            warning_id=f"accepted-warnings:malformed-entry:{rel}",
            severity="warning",
            category="accepted-warnings",
            message=(
                f"accepted-warnings file {rel!r} contains "
                f"{len(malformed)} malformed entr{'y' if len(malformed) == 1 else 'ies'}: "
                + "; ".join(malformed)
            ),
            recommended_command=None,
            evidence=[rel],
        )
    return entries, malformed_warning


# ---- detectors --------------------------------------------------------


def _detect_lock_and_journal(workspace: Workspace) -> list[DoctorWarning]:
    out: list[DoctorWarning] = []
    lock = WorkspaceLock(workspace)
    journal = OperationJournal(workspace)
    try:
        current = lock.read()
    except LockError as exc:
        out.append(
            DoctorWarning(
                warning_id="lock:malformed",
                severity="error",
                category="lock",
                message=f"workspace lock file is malformed: {exc}",
                recommended_command=(
                    "llloom status  # then llloom unlock --clear-stale "
                    "--reason \"...\" if the lock is recoverable"
                ),
                evidence=["state/locks/workspace.yaml"],
            )
        )
        current = None

    pid_state: str | None = None
    if current is not None:
        # Slice 085: classify the local owner-process state. Conservative
        # by construction — ``"alive"`` / ``"dead"`` only on confident
        # local OS evidence, ``"unknown"`` otherwise. Surfaced as
        # warning evidence and (for a same-host confidently-dead PID on
        # a live, not-yet-timed-out lock) as a dedicated warning. Never
        # governs lock clearing — the frozen timeout + journal rule is
        # the only predicate for ``is_stale_recoverable``.
        pid_state = local_owner_pid_state(current)

    if current is not None and lock.is_timed_out(current):
        recoverable, reason = lock.is_stale_recoverable(current, journal=journal)
        if recoverable:
            out.append(
                DoctorWarning(
                    warning_id=f"lock:stale-recoverable:{current.op_id}",
                    severity="warning",
                    category="lock",
                    message=(
                        f"workspace lock is timed out and recoverable "
                        f"(op_id={current.op_id}, owner={current.owner_id}, "
                        f"owner_pid_state={pid_state})"
                    ),
                    recommended_command="llloom reconcile",
                    evidence=_lock_evidence(current, pid_state),
                )
            )
        else:
            out.append(
                DoctorWarning(
                    warning_id=f"lock:stale-unrecoverable:{current.op_id}",
                    severity="error",
                    category="lock",
                    message=(
                        f"workspace lock is timed out but not recoverable "
                        f"({reason}; owner_pid_state={pid_state})"
                    ),
                    recommended_command=(
                        "llloom unlock --clear-stale --reason \"...\"  "
                        "# manual investigation may be required"
                    ),
                    evidence=_lock_evidence(current, pid_state),
                )
            )

    # Slice 085: live, not-yet-timed-out lock whose same-host owner PID
    # is confidently dead. Read-only honest signal — the stale-recovery
    # rule still requires timeout + matching in-progress journal entry,
    # so the recommended command does NOT clear the lock; it tells the
    # operator to wait through the timeout and then run reconcile or
    # unlock --clear-stale.
    if (
        current is not None
        and pid_state == PID_STATE_DEAD
        and not lock.is_timed_out(current)
    ):
        # Slice 086: the operator escape hatch
        # `llloom unlock --dead-owner --reason "..."` clears the
        # lock only when the same-host dead-owner predicate AND a
        # matching in-progress journal entry both hold. Recommend
        # that command only when the journal predicate is already
        # satisfied; otherwise keep the wording honest and point at
        # waiting / reconcile / clear-stale.
        dead_owner_journal_ok = (
            journal.exists(current.op_id)
            and journal.load(current.op_id).status == "in_progress"
            and journal.load(current.op_id).completed_at is None
        )
        if dead_owner_journal_ok:
            recommended = (
                "llloom unlock --dead-owner --reason \"...\"  "
                "# local same-host operator escape hatch (Slice 086); "
                "the journal entry is in_progress and the local owner "
                "process is confidently dead"
            )
        else:
            recommended = (
                "wait for lock to time out, then run llloom reconcile "
                "or llloom unlock --clear-stale --reason \"...\"  "
                "# guarded local recovery is not yet safe: no matching "
                "in_progress journal entry"
            )
        out.append(
            DoctorWarning(
                warning_id=f"lock:owner-process-dead:{current.op_id}",
                severity="warning",
                category="lock",
                message=(
                    f"workspace lock owner process appears dead "
                    f"(op_id={current.op_id}, owner_pid={current.owner_pid}, "
                    f"owner_hostname={current.owner_hostname}); the "
                    "ordinary stale-recovery rule still requires the "
                    "lock to time out AND its journal entry to be "
                    "in_progress before the lock may be cleared"
                ),
                recommended_command=recommended,
                evidence=_lock_evidence(current, pid_state),
            )
        )

    held_op_id = current.op_id if current is not None else None
    for entry in journal.iter_entries():
        if entry.status != "in_progress":
            continue
        if held_op_id is not None and entry.op_id == held_op_id:
            continue
        out.append(
            DoctorWarning(
                warning_id=f"lock:interrupted-journal:{entry.op_id}",
                severity="warning",
                category="lock",
                message=(
                    f"journal entry {entry.op_id} (op_kind={entry.op_kind}) "
                    "is in_progress with no matching live lock"
                ),
                recommended_command="llloom reconcile",
                evidence=[entry.op_id],
            )
        )
    return out


def _lock_evidence(current, pid_state: str | None) -> list[str]:
    """Slice 085 helper. Build the bounded evidence list for a
    lock-category warning, appending owner-process metadata only when
    present. Stable shape so accepted-warning allowlist entries that
    match on warning id alone are unaffected.
    """
    evidence: list[str] = [current.op_id, "state/locks/workspace.yaml"]
    if pid_state is not None:
        evidence.append(f"owner_pid_state={pid_state}")
    if current.owner_pid is not None:
        evidence.append(f"owner_pid={current.owner_pid}")
    if current.owner_hostname is not None:
        evidence.append(f"owner_hostname={current.owner_hostname}")
    return evidence


def _detect_transactions(workspace: Workspace) -> list[DoctorWarning]:
    """Slice 074's `state/transactions/<op_id>/` may persist past an
    interrupted render. The doctor surfaces every present transaction
    directory; ``reconcile`` is the canonical cleanup path (it removes
    transaction directories whose `op_id` matches a stale-recoverable
    lock).
    """
    out: list[DoctorWarning] = []
    root = workspace.state_transactions
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        out.append(
            DoctorWarning(
                warning_id=f"transaction:abandoned:{child.name}",
                severity="warning",
                category="transaction",
                message=(
                    f"render transaction directory {child.name} survives "
                    "after operation. Inspect manifest.yaml inside the "
                    "directory; reconcile clears matching stale locks."
                ),
                recommended_command="llloom reconcile",
                evidence=[_rel(workspace, child)],
            )
        )
    return out


def _detect_render_drift(workspace: Workspace) -> list[DoctorWarning]:
    store = ClaimStore(workspace)
    fps = FingerprintStore(workspace)
    stored = fps.load()
    recomputed = compute_page_render_fingerprints(store.iter_entities())
    drifted: set[str] = set()
    for page_id, fp in stored.items():
        if page_id not in recomputed or recomputed[page_id] != fp:
            drifted.add(page_id)
    for page_id in recomputed:
        if page_id not in stored:
            drifted.add(page_id)
    out: list[DoctorWarning] = []
    for page_id in sorted(drifted):
        out.append(
            DoctorWarning(
                warning_id=f"render:fingerprint-drift:{page_id}",
                severity="warning",
                category="render",
                message=(
                    f"render fingerprint for page {page_id!r} disagrees "
                    "with the canonical claim state"
                ),
                recommended_command="llloom reconcile",
                evidence=[page_id, "state/render_fingerprints.yaml"],
            )
        )
    return out


def _detect_sidecars(workspace: Workspace) -> list[DoctorWarning]:
    out: list[DoctorWarning] = []
    if not workspace.search_db.is_file():
        out.append(
            DoctorWarning(
                warning_id="sidecar:search:missing",
                severity="info",
                category="sidecar",
                message="optional search sidecar is missing",
                recommended_command="llloom rebuild search",
                evidence=[_rel(workspace, workspace.search_db)],
            )
        )
    if not workspace.graph_db.is_file():
        out.append(
            DoctorWarning(
                warning_id="sidecar:graph:missing",
                severity="info",
                category="sidecar",
                message="optional graph sidecar is missing",
                recommended_command="llloom rebuild graph",
                evidence=[_rel(workspace, workspace.graph_db)],
            )
        )
    return out


def _detect_structure_reports(workspace: Workspace) -> list[DoctorWarning]:
    out: list[DoctorWarning] = []
    registry = SourceRegistry(workspace)
    try:
        schema = load_schema(workspace)
    except SchemaError:
        return out
    for record in registry.iter_records():
        if record.status == "retracted":
            continue
        try:
            policy = schema.resolve_ingest_policy(record.source_class)
        except SchemaError:
            continue
        if policy != "structure_extract":
            continue
        report_path = workspace.structure_report_path(record.source_id)
        if not report_path.is_file():
            out.append(
                DoctorWarning(
                    warning_id=f"structure-report:missing:{record.source_id}",
                    severity="warning",
                    category="structure-report",
                    message=(
                        f"source {record.source_id!r} resolves to "
                        "`structure_extract` policy but has no on-disk "
                        "structure report"
                    ),
                    recommended_command=None,
                    evidence=[record.source_id, _rel(workspace, report_path)],
                )
            )
            continue
        if not _structure_report_matches(report_path, record):
            out.append(
                DoctorWarning(
                    warning_id=f"structure-report:drift:{record.source_id}",
                    severity="warning",
                    category="structure-report",
                    message=(
                        f"structure report for {record.source_id!r} no "
                        "longer matches the registered source (class / "
                        "content_hash / version mismatch)"
                    ),
                    recommended_command=None,
                    evidence=[record.source_id, _rel(workspace, report_path)],
                )
            )
    return out


def _structure_report_matches(report_path: Path, record: Any) -> bool:
    try:
        data = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("version") != "structure_report_v1":
        return False
    if data.get("source_class") != record.source_class:
        return False
    if data.get("content_hash") != record.content_hash:
        return False
    return True


def _detect_lifecycle(workspace: Workspace) -> list[DoctorWarning]:
    """Surface aggregate counts of `stale` / retracted claims so an
    operator can decide whether to clean up. These are aggregated
    signals (not per-claim warnings) because the canonical YAML
    already names the affected ids; surfacing one per claim would
    drown the doctor surface on a workspace recovering from a
    bulk retraction.
    """
    out: list[DoctorWarning] = []
    store = ClaimStore(workspace)
    stale = 0
    retracted = 0
    retracted_by_source = 0
    for entity in store.iter_entities():
        for assertion in entity.assertions:
            if assertion.status == "stale":
                stale += 1
            elif assertion.status == "retracted":
                retracted += 1
            elif assertion.status == "retracted_by_source":
                retracted_by_source += 1
    if stale:
        out.append(
            DoctorWarning(
                warning_id="lifecycle:stale-claims",
                severity="warning",
                category="lifecycle",
                message=(
                    f"{stale} claim(s) carry status='stale' (canonical "
                    "source still registered but a downstream signal "
                    "marked them stale)"
                ),
                recommended_command="llloom verify  # then promote or retract",
                evidence=[],
            )
        )
    if retracted:
        out.append(
            DoctorWarning(
                warning_id="lifecycle:retracted-claims",
                severity="info",
                category="lifecycle",
                message=(
                    f"{retracted} claim(s) carry status='retracted' "
                    "(operator-driven retraction)"
                ),
                recommended_command=None,
                evidence=[],
            )
        )
    if retracted_by_source:
        out.append(
            DoctorWarning(
                warning_id="lifecycle:retracted-by-source-claims",
                severity="info",
                category="lifecycle",
                message=(
                    f"{retracted_by_source} claim(s) carry "
                    "status='retracted_by_source' (source retraction "
                    "cascade)"
                ),
                recommended_command=None,
                evidence=[],
            )
        )
    return out


def _detect_sources(workspace: Workspace) -> list[DoctorWarning]:
    """Surface missing on-disk source files and content-hash drift
    against registered records. Hash drift is the same signal Slice
    075a refuses on the seed-apply path; surfacing it here lets the
    operator catch it before the next ingest attempt.
    """
    out: list[DoctorWarning] = []
    registry = SourceRegistry(workspace)
    for record in registry.iter_records():
        if record.status == "retracted":
            continue
        raw_path = workspace.root / record.raw_path
        if not raw_path.is_file():
            out.append(
                DoctorWarning(
                    warning_id=f"source:missing:{record.source_id}",
                    severity="error",
                    category="source",
                    message=(
                        f"raw source file for {record.source_id!r} "
                        f"({record.raw_path}) is missing on disk"
                    ),
                    recommended_command="llloom verify",
                    evidence=[record.source_id, record.raw_path],
                )
            )
            continue
        current_hash = SourceRegistry.hash_file(raw_path)
        if current_hash != record.content_hash:
            out.append(
                DoctorWarning(
                    warning_id=f"source:hash-drift:{record.source_id}",
                    severity="error",
                    category="source",
                    message=(
                        f"raw source file for {record.source_id!r} has "
                        "drifted from the registered content_hash; raw "
                        "evidence must be immutable"
                    ),
                    recommended_command=(
                        "llloom retract " + record.source_id
                        + "  # then re-ingest under a fresh source_id"
                    ),
                    evidence=[record.source_id, record.raw_path],
                )
            )
    return out


def _detect_pages(workspace: Workspace) -> list[DoctorWarning]:
    """Surface non-spine pages whose variant-(B) markers fail to
    parse. Spine pages are exempt because the editorial spine
    documents (Overview etc.) are human-authored and do not carry
    the claim-block / commentary marker contract.
    """
    out: list[DoctorWarning] = []
    pages_root = workspace.pages
    if not pages_root.is_dir():
        return out
    spine_globs = _spine_globs(workspace)
    for path in sorted(pages_root.rglob("*.md")):
        rel = _rel(workspace, path)
        if _is_spine(rel, spine_globs):
            continue
        try:
            parse_page(path.read_text(encoding="utf-8"))
        except (PageParseError, OSError) as exc:
            out.append(
                DoctorWarning(
                    warning_id=f"page:marker-parse-error:{rel}",
                    severity="warning",
                    category="page",
                    message=(
                        f"page {rel!r} has malformed variant-(B) markers: "
                        f"{exc}"
                    ),
                    recommended_command=(
                        "llloom render --dry-run  # inspect plan, then "
                        "repair the page by hand"
                    ),
                    evidence=[rel],
                )
            )
    return out


def _spine_globs(workspace: Workspace) -> set[str]:
    """Read `schema/spine_manifest.yaml` to learn which pages are
    editorial-spine pages (those don't carry the marker contract).
    Returns a set of workspace-relative POSIX paths and glob patterns.
    Errors are swallowed — a malformed spine manifest is not the
    doctor's problem to solve.
    """
    manifest_path = workspace.root / "schema" / "spine_manifest.yaml"
    out: set[str] = set()
    if not manifest_path.is_file():
        return out
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return out
    for key in ("spine_files", "spine_globs"):
        for value in data.get(key) or []:
            if isinstance(value, str):
                out.add(value)
    return out


def _is_spine(rel: str, spine: Iterable[str]) -> bool:
    for pattern in spine:
        if pattern == rel:
            return True
        # Slice 079: keep the glob match simple — only the trailing
        # ``/**`` form needs special handling because that's what the
        # default starter manifest uses for ``pages/navigation/**``.
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            if rel == prefix or rel.startswith(prefix + "/"):
                return True
    return False


# ---- update review bundle --------------------------------------------


def _build_update_review_bundle(
    workspace: Workspace,
    *,
    op_id: str | None,
    last_op: bool,
    warnings: list[DoctorWarning],
    accepted: list[AcceptedDoctorWarning],
) -> UpdateReviewBundle | None:
    journal = OperationJournal(workspace)
    if last_op:
        entry = journal.latest()
        if entry is None:
            warnings.append(
                DoctorWarning(
                    warning_id="review-bundle:no-journal-entries",
                    severity="warning",
                    category="review-bundle",
                    message=(
                        "doctor --last-op requested but the journal "
                        "directory has no entries"
                    ),
                    recommended_command=None,
                    evidence=[],
                )
            )
            return None
        resolved_op_id = entry.op_id
    else:
        assert op_id is not None  # caller branch guarantee
        if not journal.exists(op_id):
            warnings.append(
                DoctorWarning(
                    warning_id=f"review-bundle:unknown-op-id:{op_id}",
                    severity="warning",
                    category="review-bundle",
                    message=(
                        f"doctor --op-id {op_id!r} refers to a journal "
                        "entry that does not exist"
                    ),
                    recommended_command=None,
                    evidence=[op_id],
                )
            )
            return None
        entry = journal.load(op_id)
        resolved_op_id = op_id

    # Seed update report.
    seed_report_rel: str | None = None
    seed_data: dict[str, Any] | None = None
    seed_report_path = (
        workspace.state_reports_updates / f"{resolved_op_id}.yaml"
    )
    if seed_report_path.is_file():
        try:
            seed_data = yaml.safe_load(
                seed_report_path.read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError):
            seed_data = None
        if isinstance(seed_data, dict):
            seed_report_rel = _rel(workspace, seed_report_path)
        else:
            seed_data = None

    source_changes: list[str] = []
    claim_changes: list[str] = []
    rendered_pages: list[str] = []
    provenance: dict[str, Any] = {}
    if isinstance(seed_data, dict):
        for src in seed_data.get("sources") or []:
            if isinstance(src, dict) and isinstance(src.get("source_id"), str):
                source_changes.append(src["source_id"])
        for created in seed_data.get("claims_created") or []:
            if isinstance(created, dict) and isinstance(
                created.get("claim_id"), str
            ):
                claim_changes.append(created["claim_id"])
        for page in seed_data.get("pages", {}).get("rendered") or []:
            if isinstance(page, str):
                rendered_pages.append(page)
        prov = seed_data.get("provenance")
        if isinstance(prov, dict):
            provenance = dict(prov)

    # Lint, verify, status summaries (each read-only).
    lint_summary = _lint_summary(workspace)
    verify_summary = _verify_summary(workspace)
    status_summary = _status_summary(workspace)

    return UpdateReviewBundle(
        op_id=resolved_op_id,
        op_kind=entry.op_kind,
        journal_status=entry.status,
        started_at=entry.started_at,
        completed_at=entry.completed_at,
        touched_files=list(entry.touched_files),
        seed_update_report_path=seed_report_rel,
        source_changes=source_changes,
        claim_changes=claim_changes,
        rendered_pages=rendered_pages,
        lint_summary=lint_summary,
        verify_summary=verify_summary,
        status_summary=status_summary,
        provenance=provenance,
        warnings=list(warnings),
        accepted_warnings=list(accepted),
    )


def _lint_summary(workspace: Workspace) -> dict[str, int]:
    result = run_lint(workspace)
    return {
        "failures": len(result.failures),
        "warnings": len(result.warnings),
        "canary_hits": len(result.canary_hits),
    }


def _verify_summary(workspace: Workspace) -> dict[str, int]:
    result = run_verify(workspace)
    return {
        "verified": len(result.verified),
        "failed": len(result.failed),
        "mismatches": len(result.mismatches),
    }


def _status_summary(workspace: Workspace) -> dict[str, Any]:
    result = run_status(workspace)
    return {
        "source_count": result.source_count,
        "claim_count": result.claim_count,
        "rendered_page_count": result.rendered_page_count,
        "pending_review_count": result.pending_review_count,
        "stale_count": result.stale_count,
        "retracted_count": result.retracted_count,
        "lock_held": result.lock_held,
        "lock_owner": result.lock_owner,
        "lock_is_timed_out": result.lock_is_timed_out,
        "lock_recoverable": result.lock_recoverable,
    }


# ---- helpers ----------------------------------------------------------


_SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}


def _warning_sort_key(w: DoctorWarning) -> tuple[int, str, str]:
    return (_SEVERITY_RANK.get(w.severity, 99), w.category, w.warning_id)


def _aggregate_recommendations(warnings: list[DoctorWarning]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for w in warnings:
        if w.recommended_command is None:
            continue
        if w.recommended_command in seen:
            continue
        seen.add(w.recommended_command)
        out.append(w.recommended_command)
    return out


def _rel(workspace: Workspace, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
