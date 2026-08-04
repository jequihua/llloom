"""Command-line interface for the first vertical slice.

See ``04_specification/operations_and_cli.md`` Â§"CLI shape" for the
frozen verb set.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from llloom.llm.harness import LLMInvoke
from llloom.ops.page import CANDIDATE_FRAMEWORK_PROFILE
from llloom.ops import (
    ClaimCardError,
    InspectFilterError,
    PageCreateError,
    SupersedeError,
    apply_seed_manifest,
    claim_card,
    create_page,
    doctor,
    ingest,
    lint,
    list_claims,
    list_merge_proposals,
    list_pages,
    list_render_targets,
    list_sources,
    merge_alias,
    prepare_pdf,
    promote,
    query,
    rebuild,
    reconcile,
    reject_alias,
    render,
    retract,
    review_alias,
    status,
    supersede,
    unlock,
    verify,
)
from llloom.ops.query import QueryFilterError
from llloom.pages.regions import PageParseError
from llloom.pages.render import RenderError
from llloom.workspace.layout import Workspace


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        ws = Workspace.init(args.root)
        _print(
            {
                "command": "init",
                "root": str(ws.root),
            }
        )
        return 0

    ws = Workspace.load(args.root)

    if args.command == "status":
        _print(asdict(status(ws)))
        return 0

    if args.command == "ingest":
        # Slice 081: resolve relative source paths against
        # ``workspace.root`` so the CLI honors ``--root`` consistently.
        # Absolute paths stay absolute. The pre-Slice-081 behavior
        # (cwd-relative resolution) is silently preserved when the
        # operator runs from inside the memory root because
        # ``cwd == workspace.root`` makes the two resolutions agree.
        raw_path = Path(args.path)
        if raw_path.is_absolute():
            src_path = raw_path.resolve()
        else:
            src_path = (ws.root / raw_path).resolve()
        if not src_path.is_file():
            sys.stderr.write(
                f"llloom ingest: source path not found: {src_path} "
                f"(workspace root: {ws.root})\n"
            )
            return 1
        try:
            harness = _build_harness(args)
        except _CLIError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        result = ingest(
            ws,
            src_path,
            source_id=args.source_id,
            source_class=args.source_class,
            no_render=args.no_render,
            llm=harness,
            structure_source_ids=list(args.structure_source or []),
        )
        _print(_dc(result))
        return 0 if result.succeeded else 1

    if args.command == "verify":
        result = verify(ws, target=args.target)
        _print(_dc(result))
        return 0 if result.passed else 1

    if args.command == "render":
        try:
            result = render(
                ws,
                target=args.target,
                dry_run=bool(args.dry_run),
                list_targets=bool(args.list_targets),
            )
        except ValueError as exc:
            sys.stderr.write(f"llloom render: {exc}\n")
            return 1
        except (PageParseError, RenderError) as exc:
            # Slice 081: expected operational marker / render failures
            # surface as concise stderr instead of a Python traceback.
            # Library exception behavior is unchanged.
            sys.stderr.write(f"llloom render: {exc}\n")
            sys.stderr.write(
                "next: fix the page markers, then run "
                "`llloom render --dry-run`\n"
            )
            return 1
        _print(_dc(result))
        return 0

    if args.command == "query":
        try:
            result = query(
                ws,
                question=args.question,
                status=args.status or None,
                verification_status=args.verification_status,
                entity_id=args.entity,
                role=args.role,
                ids_only=bool(args.ids_only),
            )
        except QueryFilterError as exc:
            sys.stderr.write(f"llloom query: {exc}\n")
            return 1
        if args.ids_only:
            _print_ids(result.used_claim_ids)
        else:
            _print(_dc(result))
        return 0

    if args.command == "list-claims":
        try:
            summaries = list_claims(
                ws,
                status=args.status or None,
                verification_status=args.verification_status,
                entity_id=args.entity,
                role=args.role,
            )
        except InspectFilterError as exc:
            sys.stderr.write(f"llloom list-claims: {exc}\n")
            return 1
        if args.ids_only:
            _print_ids([s.qualified_target for s in summaries])
        else:
            _print(_dc(summaries))
        return 0

    if args.command == "claim-card":
        try:
            card = claim_card(ws, args.target)
        except ClaimCardError as exc:
            sys.stderr.write(f"llloom claim-card: {exc}\n")
            return 1
        _print(_dc(card))
        return 0

    if args.command == "list-sources":
        try:
            sources = list_sources(
                ws,
                status=args.status or None,
                source_class=args.source_class,
            )
        except InspectFilterError as exc:
            sys.stderr.write(f"llloom list-sources: {exc}\n")
            return 1
        if args.ids_only:
            _print_ids([s.source_id for s in sources])
        else:
            _print(_dc(sources))
        return 0

    if args.command == "list-pages":
        pages = list_pages(ws)
        if args.ids_only:
            _print_ids([p.page_id for p in pages])
        else:
            _print(_dc(pages))
        return 0

    if args.command == "list-render-targets":
        targets = list_render_targets(ws, page=args.page)
        if args.ids_only:
            _print_ids([t.page_id for t in targets])
        else:
            _print(_dc(targets))
        return 0

    if args.command == "lint":
        result = lint(ws, generated_canary=args.generated_canary)
        _print(_dc(result))
        return 0 if result.passed else 1

    if args.command == "reconcile":
        _print(_dc(reconcile(ws)))
        return 0

    if args.command == "unlock":
        if args.clear_stale and args.dead_owner:
            sys.stderr.write(
                "llloom unlock: pass at most one of --clear-stale "
                "/ --dead-owner\n"
            )
            return 1
        # Slice 086a: --dead-owner does not accept a positional
        # target — the mode operates only on the symbolic workspace
        # target. Refuse cleanly at the CLI layer before the
        # library is reached.
        if args.dead_owner and isinstance(args.target, str) and args.target:
            sys.stderr.write(
                "llloom unlock: --dead-owner does not accept a target; "
                "omit the positional target (the symbolic target is "
                "always 'workspace')\n"
            )
            return 1
        result = unlock(
            ws,
            target=args.target,
            reason=args.reason,
            clear_stale=bool(args.clear_stale),
            dead_owner=bool(args.dead_owner),
        )
        _print(_dc(result))
        return 1 if result.refused else 0

    if args.command == "promote":
        result = promote(ws, target=args.target, to_status=args.to)
        _print(_dc(result))
        return 0 if not result.refused else 1

    if args.command == "retract":
        _print(_dc(retract(ws, source_id=args.source_id, reason=args.reason)))
        return 0

    if args.command == "supersede":
        try:
            result = supersede(ws, old=args.old, by=args.by)
        except SupersedeError as exc:
            sys.stderr.write(f"llloom supersede: {exc}\n")
            return 1
        _print(_dc(result))
        return 0 if result.succeeded else 1

    if args.command == "doctor":
        if args.op_id is not None and args.last_op:
            sys.stderr.write(
                "llloom doctor: pass at most one of --op-id / --last-op\n"
            )
            return 1
        try:
            result = doctor(
                ws,
                op_id=args.op_id,
                last_op=bool(args.last_op),
                accepted_warnings=args.accepted_warnings,
            )
        except ValueError as exc:
            sys.stderr.write(f"llloom doctor: {exc}\n")
            return 1
        _print(_dc(result))
        # Doctor warnings are diagnostic, not command failure. Exit 1
        # only when an unaccepted ``error`` severity is present, or
        # when a requested op id / last_op selection refused.
        has_error = any(w.severity == "error" for w in result.warnings)
        has_review_refusal = any(
            w.category == "review-bundle" for w in result.warnings
        )
        return 1 if has_error or has_review_refusal else 0

    if args.command == "rebuild":
        _print(rebuild(ws, target=args.target))
        return 0

    if args.command == "list_merge_proposals":
        _print([_dc(p) for p in list_merge_proposals(ws)])
        return 0

    if args.command == "review-alias":
        _print(
            _dc(
                review_alias(
                    ws,
                    proposal_id=args.proposal_id,
                    decision=args.decision,
                    notes=args.notes,
                )
            )
        )
        return 0

    if args.command == "merge-alias":
        _print(_dc(merge_alias(ws, proposal_id=args.proposal_id)))
        return 0

    if args.command == "reject-alias":
        _print(_dc(reject_alias(ws, proposal_id=args.proposal_id, notes=args.notes)))
        return 0

    if args.command == "seed":
        if args.seed_subcommand != "apply":
            parser.error(f"unknown seed subcommand {args.seed_subcommand!r}")
        result = apply_seed_manifest(
            ws,
            args.manifest_path,
            dry_run=bool(args.dry_run),
            no_render=bool(args.no_render),
            status=args.status,
        )
        _print(_dc(result))
        return 0 if result.succeeded else 1

    if args.command == "prepare-pdf":
        result = prepare_pdf(
            ws,
            pdf_path=Path(args.pdf_path),
            prep_id=args.prep_id,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        _print(_dc(result))
        return 0 if result.succeeded else 1

    if args.command == "page":
        if args.page_subcommand != "create":
            parser.error(
                f"unknown page subcommand {args.page_subcommand!r}"
            )
        try:
            result = create_page(
                ws,
                page_id=args.page_id,
                page_class=args.page_class,
                title=args.title,
                framework_profile=args.framework_profile,
            )
        except PageCreateError as exc:
            sys.stderr.write(f"llloom page create: {exc}\n")
            return 1
        _print(_dc(result))
        return 0 if result.succeeded else 1

    parser.error(f"unknown command {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llloom")
    parser.add_argument(
        "--root",
        default=".",
        help="workspace root (default: current working directory)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="initialize or validate a workspace")
    sub.add_parser("status", help="compact workspace health summary")

    p = sub.add_parser("ingest", help="register and (per policy) ingest a source")
    p.add_argument("path", help="path to raw source file under raw/")
    p.add_argument("--source-id", dest="source_id", default=None)
    p.add_argument("--source-class", dest="source_class", default=None)
    p.add_argument(
        "--no-render",
        dest="no_render",
        action="store_true",
        default=False,
        help=(
            "suppress the render step for claim_extract_and_view_render "
            "ingests; verified claims still persist, IngestResult.render_skipped "
            "is true. Has no effect on other policies."
        ),
    )
    p.add_argument(
        "--model-provider",
        dest="model_provider",
        default=None,
        choices=["openai", "anthropic"],
        help=(
            "optional provider adapter for claim_extract ingestion. "
            "Default behavior (flag absent) is unchanged and offline: "
            "NullModel produces zero candidates. 'openai' requires the "
            "optional extra: pip install \"llloom[openai]\". "
            "'anthropic' requires the optional extra: "
            "pip install \"llloom[anthropic]\"."
        ),
    )
    p.add_argument(
        "--model",
        dest="model_id",
        default=None,
        help=(
            "provider model id (e.g. gpt-5.4 for openai, "
            "claude-sonnet-4-5-20250929 for anthropic). Required with "
            "--model-provider."
        ),
    )
    p.add_argument(
        "--model-timeout",
        dest="model_timeout",
        default=None,
        type=float,
        help="optional request timeout in seconds for the provider adapter.",
    )
    p.add_argument(
        "--structure-source",
        dest="structure_source",
        action="append",
        default=None,
        help=(
            "explicit registered structure-source id whose on-disk "
            "structure report should be passed to the provider as "
            "metadata-only context. Repeatable. Only "
            "claim_extract / claim_extract_and_view_render consume "
            "the value; index_only / structure_extract / deny ignore "
            "it. Missing or stale reports refuse the ingest cleanly."
        ),
    )

    p = sub.add_parser("verify", help="run deterministic provenance verification")
    p.add_argument("target", nargs="?", default=None)

    p = sub.add_parser("render", help="render claim-block regions")
    p.add_argument("target", nargs="?", default=None)
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help=(
            "compute the render plan without writing any page or "
            "fingerprint and without acquiring the workspace lock or "
            "opening a render journal entry. The result's `plan` field "
            "lists contributors, claim ids, marker health, and "
            "would-change flags per page."
        ),
    )
    p.add_argument(
        "--list-targets",
        dest="list_targets",
        action="store_true",
        default=False,
        help=(
            "enumerate valid render targets with contributor and "
            "marker details. Read-only: no lock, no journal, no page "
            "or fingerprint write. A `page:<id>` whose page exists on "
            "disk but has no contributing claims still appears with "
            "an empty contributors list."
        ),
    )

    p = sub.add_parser("query", help="read-only authoritative query")
    p.add_argument("question")
    p.add_argument(
        "--status",
        dest="status",
        action="append",
        default=None,
        help=(
            "filter by lifecycle status. Repeatable. Pass 'all' to "
            "include every state (including retracted/superseded/"
            "archived/stale/retracted_by_source). Omit for the "
            "default behavior (skip retracted/retracted_by_source/"
            "archived/stale/superseded). Slice 077: introduced as a "
            "read-only AND constraint on top of the canonical "
            "default-broad behavior, never widens visible state."
        ),
    )
    p.add_argument(
        "--verification-status",
        dest="verification_status",
        default=None,
        choices=sorted({"unverified", "verified", "failed"}),
        help="filter by verification status.",
    )
    p.add_argument(
        "--entity",
        dest="entity",
        default=None,
        help="filter to claims on this entity_id.",
    )
    p.add_argument(
        "--role",
        dest="role",
        default=None,
        help=(
            "filter by claim role/kind (matches Assertion.claim_kind). "
            "Slice 077: claim_kind is the V1 role/kind field; the "
            "codebase has no separate role attribute."
        ),
    )
    p.add_argument(
        "--ids-only",
        dest="ids_only",
        action="store_true",
        default=False,
        help=(
            "emit one claim_id per line with no answer prose, no JSON "
            "envelope, and no bullet markers. Suitable for prompts, "
            "reviews, and shell pipelines."
        ),
    )

    # Slice 077: read-only listing / card surfaces.
    p = sub.add_parser(
        "list-claims",
        help=(
            "read-only filtered claim listing (Slice 077). Walks "
            "canonical claim YAML and reports a one-line summary "
            "per claim. No lock, no journal, no model call."
        ),
    )
    p.add_argument(
        "--status",
        dest="status",
        action="append",
        default=None,
        help=(
            "filter by lifecycle status. Repeatable. Pass 'all' "
            "explicitly to be unambiguous (the default already shows "
            "every state for listing surfaces)."
        ),
    )
    p.add_argument(
        "--verification-status",
        dest="verification_status",
        default=None,
        choices=sorted({"unverified", "verified", "failed"}),
    )
    p.add_argument("--entity", dest="entity", default=None)
    p.add_argument(
        "--role",
        dest="role",
        default=None,
        help=(
            "filter by claim role/kind (matches Assertion.claim_kind). "
            "Slice 077: claim_kind is the V1 role/kind field."
        ),
    )
    p.add_argument(
        "--ids-only",
        dest="ids_only",
        action="store_true",
        default=False,
        help=(
            "emit one qualified target (claim:<entity>:<claim>) per "
            "line, no JSON envelope. Suitable for shell pipelines."
        ),
    )

    p = sub.add_parser(
        "claim-card",
        help=(
            "read-only inspection of one claim (Slice 077). Accepts "
            "either claim:<entity_id>:<claim_id> or a bare claim_id "
            "(only when unique). Ambiguous bare ids refuse with "
            "candidate qualified ids."
        ),
    )
    p.add_argument(
        "target",
        help=(
            "claim:<entity_id>:<claim_id> or bare claim_id (when "
            "unique across the workspace)."
        ),
    )

    p = sub.add_parser(
        "list-sources",
        help=(
            "read-only source registry listing (Slice 077). Reports "
            "registry metadata only; never reads raw source bodies."
        ),
    )
    p.add_argument(
        "--status",
        dest="status",
        action="append",
        default=None,
        help=(
            "filter by registry status (registered/retracted). "
            "Repeatable."
        ),
    )
    p.add_argument(
        "--source-class",
        dest="source_class",
        default=None,
        help="filter by source_class string.",
    )
    p.add_argument(
        "--ids-only",
        dest="ids_only",
        action="store_true",
        default=False,
        help="emit one source_id per line, no JSON envelope.",
    )

    p = sub.add_parser(
        "list-pages",
        help=(
            "read-only page tree listing (Slice 077). Walks pages/ "
            "and reports each page's id, path, and frontmatter "
            "fields."
        ),
    )
    p.add_argument(
        "--ids-only",
        dest="ids_only",
        action="store_true",
        default=False,
        help="emit one page_id per line, no JSON envelope.",
    )

    p = sub.add_parser(
        "list-render-targets",
        help=(
            "read-only render-target enumeration (Slice 077). Reuses "
            "the Slice 073 read-only render plan; no lock, no "
            "journal, no page/fingerprint write."
        ),
    )
    p.add_argument(
        "--page",
        dest="page",
        default=None,
        help="limit to one page_id.",
    )
    p.add_argument(
        "--ids-only",
        dest="ids_only",
        action="store_true",
        default=False,
        help="emit one page_id per line, no JSON envelope.",
    )

    p = sub.add_parser("lint", help="workspace health checks")
    p.add_argument(
        "--generated-canary",
        dest="generated_canary",
        action="store_true",
        default=False,
        help=(
            "include a fresh high-entropy per-run canary token in the "
            "canary scan set. The fixed fixture token is always "
            "scanned; this flag adds release-validation coverage. "
            "Clean workspaces still pass."
        ),
    )
    sub.add_parser("reconcile", help="repair interrupted state")

    p = sub.add_parser(
        "unlock",
        help=(
            "bare: record a time-bounded maintenance unlock window "
            "(journal only, does not clear the workspace lock); "
            "with --clear-stale: clear a journal-backed stale "
            "workspace lock (refuses live, completed-journal, or "
            "missing-journal locks)"
        ),
    )
    p.add_argument(
        "target",
        nargs="?",
        default=None,
        help=(
            "required for the bare unlock-window mode; omit when "
            "using --clear-stale (the symbolic target is recorded "
            "as 'workspace')"
        ),
    )
    p.add_argument("--reason", required=True)
    p.add_argument(
        "--clear-stale",
        dest="clear_stale",
        action="store_true",
        default=False,
        help=(
            "clear the workspace lock file when "
            "WorkspaceLock.is_stale_recoverable(...) returns true "
            "(timed-out lock + matching in-progress journal). The "
            "prior journal entry is marked interrupted and a "
            "completed audit entry (op_kind=unlock_clear_stale) is "
            "written. Refuses live locks, completed-journal locks, "
            "and missing-journal locks; never deletes the lock file "
            "on the refusal path."
        ),
    )
    p.add_argument(
        "--dead-owner",
        dest="dead_owner",
        action="store_true",
        default=False,
        help=(
            "Slice 086 — clear the workspace lock when the local "
            "same-host owner process is confidently dead AND the "
            "matching journal entry is still in_progress. Mutually "
            "exclusive with --clear-stale. Requires lock owner "
            "metadata (owner_pid + owner_hostname) and a non-empty "
            "--reason. Refuses remote-host locks, missing-metadata "
            "locks, alive/unknown PID locks, timed-out locks (use "
            "--clear-stale for those), missing/completed-journal "
            "locks, and any lock whose identity changes during a "
            "final pre-clear re-read. Audit entry op_kind="
            "unlock_clear_dead_owner."
        ),
    )

    p = sub.add_parser("promote", help="advance lifecycle for claim:<entity>:<id>")
    p.add_argument("target")
    p.add_argument("--to", required=True, choices=["reviewed", "validated", "superseded", "archived"])

    p = sub.add_parser("retract", help="retract a source")
    p.add_argument("source_id")
    p.add_argument("--reason", default=None)

    p = sub.add_parser(
        "doctor",
        help=(
            "read-only workspace diagnostic (Slice 079). Detects "
            "stale lock, interrupted journal, abandoned render "
            "transactions, render fingerprint drift, missing "
            "sidecars, missing / stale structure reports, lifecycle "
            "/ source / page anomalies, and emits stable warning ids "
            "with recommended next commands. Optional accepted-"
            "warning allowlist separates known signals from new "
            "ones. Pass --op-id or --last-op to additionally build a "
            "memory-update review bundle for one operation. Never "
            "writes anything: no lock, journal, transaction, page, "
            "fingerprint, claim, source, sidecar, or report write."
        ),
    )
    p.add_argument(
        "--op-id",
        dest="op_id",
        default=None,
        help=(
            "build an UpdateReviewBundle for this journal op id. "
            "Mutually exclusive with --last-op."
        ),
    )
    p.add_argument(
        "--last-op",
        dest="last_op",
        action="store_true",
        default=False,
        help=(
            "build an UpdateReviewBundle for the most recent journal "
            "entry. Mutually exclusive with --op-id."
        ),
    )
    p.add_argument(
        "--accepted-warnings",
        dest="accepted_warnings",
        default=None,
        help=(
            "path to an accepted-warnings allowlist (YAML, "
            "accepted_warnings_v1). Defaults to "
            "state/reports/health/accepted_warnings.yaml when "
            "present; never created by doctor."
        ),
    )

    p = sub.add_parser(
        "supersede",
        help=(
            "direct lifecycle supersede (Slice 078). Marks OLD as "
            "'superseded' and records a supersession link from --by "
            "back to OLD on the canonical Assertion.supersedes "
            "field. Both targets must already be 'validated'; the "
            "operation is atomic under the existing operation(...) "
            "lock / journal contract."
        ),
    )
    p.add_argument(
        "old",
        help=(
            "old claim target: 'claim:<entity_id>:<claim_id>' or a "
            "bare claim_id (when unique across the workspace)."
        ),
    )
    p.add_argument(
        "--by",
        dest="by",
        required=True,
        help=(
            "new (replacement) claim target. Same grammar as the "
            "positional OLD argument."
        ),
    )

    p = sub.add_parser("rebuild", help="recompute a named derived artifact")
    p.add_argument(
        "target",
        choices=["render_fingerprints", "health_report", "index", "log", "search", "graph"],
    )

    sub.add_parser("list_merge_proposals", help="list pending alias merge proposals")

    p = sub.add_parser("review-alias", help="approve/reject an alias proposal")
    p.add_argument("proposal_id")
    p.add_argument("--decision", required=True, choices=["approve", "reject"])
    p.add_argument("--notes", default=None)

    p = sub.add_parser("merge-alias", help="apply an approved alias proposal")
    p.add_argument("proposal_id")

    p = sub.add_parser("reject-alias", help="close an alias proposal as rejected")
    p.add_argument("proposal_id")
    p.add_argument("--notes", default=None)

    p = sub.add_parser(
        "seed",
        help=(
            "deterministic seed-manifest application (Slice 075). "
            "Subcommands: apply <manifest.yaml>. Never invokes a "
            "model; manifest claims flow through the existing verifier "
            "and atomic claim-store primitives."
        ),
    )
    seed_sub = p.add_subparsers(dest="seed_subcommand", required=True)
    apply_p = seed_sub.add_parser(
        "apply",
        help=(
            "apply a seed_manifest_v1 YAML to the workspace. "
            "Persists every verified claim through the existing "
            "batch-atomic path."
        ),
    )
    apply_p.add_argument(
        "manifest_path",
        help="path to the seed_manifest_v1 YAML (workspace-relative or absolute)",
    )
    apply_p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help=(
            "validate the manifest and verifier-walk every claim "
            "without writing anything (no source registry update, "
            "no claim YAML, no page render, no fingerprint, no "
            "journal entry). Result reports planned sources + "
            "claims with their effective merged status."
        ),
    )
    apply_p.add_argument(
        "--no-render",
        dest="no_render",
        action="store_true",
        default=False,
        help=(
            "persist verified claims but suppress the post-apply "
            "page render. Render targets remain on the persisted "
            "claims; the next `llloom render` (or any future apply "
            "without --no-render) catches the affected pages up."
        ),
    )
    apply_p.add_argument(
        "--status",
        dest="status",
        default=None,
        help=(
            "operation-level default lifecycle status for claims "
            "whose merged status is unset. Validated against "
            "CLAIM_STATUSES before any persistence; per-claim "
            "explicit status still wins."
        ),
    )

    p = sub.add_parser(
        "prepare-pdf",
        help=(
            "optional: prepare a PDF into a working-text bundle "
            "(requires the llloom[docling] extra)"
        ),
    )
    p.add_argument("pdf_path", help="path to the input PDF file")
    p.add_argument(
        "--prep-id",
        dest="prep_id",
        default=None,
        help=(
            "stable bundle id. If omitted, derived from the PDF "
            "filename stem. Only [A-Za-z0-9._-] are accepted."
        ),
    )
    p.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help=(
            "workspace-relative output root for prep bundles "
            "(default: raw/derived/pdf)."
        ),
    )
    p.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="replace an existing bundle at the target prep-id.",
    )

    p = sub.add_parser(
        "page",
        help=(
            "page-level mutating operations (Slice 084). Subcommands: "
            "create <page_id> writes one valid variant-(B) page stub "
            "under pages/<class_dir>/<tail>.md. No overwrite flag; "
            "refuses existing pages."
        ),
    )
    page_sub = p.add_subparsers(dest="page_subcommand", required=True)
    create_p = page_sub.add_parser(
        "create",
        help=(
            "create one valid variant-(B) page stub. Creates no "
            "claims, invokes no model, performs no render and no "
            "fingerprint update. Journaled mutating operation under "
            "the existing operation(...) lock contract."
        ),
    )
    create_p.add_argument(
        "page_id",
        help=(
            "page id (e.g. 'concept/foo' or 'foo'). Forward-slash "
            "separators only. Must not end in '.md'. The first "
            "segment may be a recognized page-class prefix "
            "(entity/concept/synthesis/navigation or the plural "
            "directory form) — when present, the class is inferred "
            "and --page-class may be omitted."
        ),
    )
    create_p.add_argument(
        "--page-class",
        dest="page_class",
        default=None,
        choices=["entity", "concept", "synthesis", "navigation"],
        help=(
            "explicit page class. Required when the page_id does "
            "not start with a recognized class prefix. If both an "
            "explicit class and an inferred prefix are present, "
            "they must match."
        ),
    )
    create_p.add_argument(
        "--title",
        dest="title",
        default=None,
        help=(
            "Markdown H1 title for the stub. Defaults to a "
            "human-readable title derived from the final page_id "
            "segment (e.g. 'foo-bar' -> 'Foo Bar')."
        ),
    )
    create_p.add_argument(
        "--framework-profile",
        dest="framework_profile",
        default=None,
        choices=[CANDIDATE_FRAMEWORK_PROFILE],
        help=(
            "explicit per-request opt-in (Slice M003/S01). When "
            "supplied, the created page carries the two-field "
            "candidate producer minimum and is written with LF "
            "bytes on every platform. Omitting it is the legacy "
            "path and is byte-frozen. Grants no verification, "
            "lifecycle, render, or execution authority."
        ),
    )

    return parser


class _CLIError(Exception):
    """Raised by CLI-level plumbing (not by the library)."""


def _build_harness(args) -> LLMInvoke | None:
    """Construct an ``LLMInvoke`` when the caller asked for a provider.

    Returns ``None`` for the default (offline) case so ``ingest``
    stays on its existing ``NullModel`` path. The library API surface
    is unchanged — this is purely CLI plumbing.
    """
    provider = getattr(args, "model_provider", None)
    if provider is None:
        return None
    if provider == "openai":
        if not args.model_id:
            raise _CLIError(
                "--model-provider openai requires --model <model-id>"
            )
        try:
            import openai  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise _CLIError(
                "the OpenAI optional dependency is not installed; "
                "install it with: pip install \"llloom[openai]\""
            ) from exc
        from llloom.llm.openai_backend import OpenAIModelBackend

        backend: Any = OpenAIModelBackend(
            model=args.model_id,
            timeout=args.model_timeout,
        )
        return LLMInvoke(model=backend)
    if provider == "anthropic":
        if not args.model_id:
            raise _CLIError(
                "--model-provider anthropic requires --model <model-id>"
            )
        try:
            import anthropic  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise _CLIError(
                "the Anthropic optional dependency is not installed; "
                "install it with: pip install \"llloom[anthropic]\""
            ) from exc
        from llloom.llm.anthropic_backend import AnthropicModelBackend

        backend = AnthropicModelBackend(
            model=args.model_id,
            timeout=args.model_timeout,
        )
        return LLMInvoke(model=backend)
    raise _CLIError(f"unknown model provider {provider!r}")


def _print(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, default=str, sort_keys=False))
    sys.stdout.write("\n")


def _print_ids(ids: list[str]) -> None:
    """Slice 077: emit one id per line with no JSON envelope, no
    bullets, and no trailing commentary. Suitable for shell pipelines.
    """
    for value in ids:
        sys.stdout.write(value)
        sys.stdout.write("\n")


def _dc(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, list):
        return [_dc(x) for x in obj]
    return obj


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

