"""``page create`` operation (Slice 084).

Deterministic creation of a single valid variant-(B) page stub.

The operation:

- runs under the existing ``operation(...)`` lock + journal contract;
- writes exactly one Markdown file under ``pages/<class_dir>/<tail>.md``;
- creates no claims, invokes no model, performs no render and no
  render-fingerprint update;
- refuses cleanly (no traceback) on malformed page ids, unknown page
  classes, conflicting inferred / explicit classes, existing target
  files, and any path that would escape the selected class directory;
- runs the generated text through :func:`llloom.pages.regions.parse_page`
  before writing so a malformed internal template fails before any
  page becomes visible.

The slice intentionally does not add an overwrite flag — operators
must remove the existing page (or use a future page-edit verb) before
re-creating. Pre-existing pages refuse before the operation context
opens.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from llloom.okf import observe_page_okf
from llloom.ops._context import operation, relative_posix
from llloom.ops.results import PageCreateResult
from llloom.pages.regions import PageParseError, parse_page
from llloom.workspace.layout import Workspace

#: The single accepted ``framework_profile`` opt-in value (M003/S01).
CANDIDATE_FRAMEWORK_PROFILE = "0.1-rc.1"


class PageCreateError(Exception):
    """Raised by ``create_page(...)`` for pre-operation refusals.

    The CLI catches this and prints a concise ``llloom page create:
    <message>`` diagnostic with exit code 1. Refusal modes inside the
    operation context (e.g. a target file that appeared after the
    pre-check) surface via :class:`PageCreateResult.refusal_reason`
    so they still carry an ``op_id`` for journal correlation.
    """


_PAGE_CLASSES: dict[str, str] = {
    "entity": "entities",
    "concept": "concepts",
    "synthesis": "syntheses",
    "navigation": "navigation",
}

_CLASS_PREFIXES: dict[str, str] = {
    "entity": "entity",
    "entities": "entity",
    "concept": "concept",
    "concepts": "concept",
    "synthesis": "synthesis",
    "syntheses": "synthesis",
    "navigation": "navigation",
}

_PAGE_ID_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_MARKER_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def create_page(
    workspace: Workspace,
    *,
    page_id: str,
    page_class: str | None = None,
    title: str | None = None,
    framework_profile: str | None = None,
) -> PageCreateResult:
    """Create one valid variant-(B) page stub under ``pages/``.

    Raises :class:`PageCreateError` for malformed-argument or
    pre-existing-target refusals so the CLI can emit a concise
    diagnostic with no traceback. Returns a
    :class:`PageCreateResult` with ``refusal_reason`` set when a
    refusal is detected inside the operation context (e.g. a target
    that appeared between the pre-check and the lock acquisition).

    ``framework_profile`` is the M003/S01 explicit per-request opt-in.
    Omitting it is the legacy path and is byte-frozen, including the
    accepted host-dependent newline translation of the native writer.
    Passing exactly :data:`CANDIDATE_FRAMEWORK_PROFILE` adds the two
    producer-minimum frontmatter fields and writes LF bytes on every
    platform. Any other value refuses before an operation opens.
    """
    _validate_framework_profile(framework_profile)
    normalized_page_id, inferred_class = _validate_page_id(page_id)
    resolved_class = _resolve_page_class(
        page_class=page_class, inferred_class=inferred_class
    )
    page_path = _resolve_page_path(
        workspace, normalized_page_id=normalized_page_id, page_class=resolved_class
    )

    if page_path.exists():
        raise PageCreateError(
            f"page already exists at {relative_posix(workspace, page_path)!r}; "
            "remove it before re-creating (no overwrite flag in this slice)"
        )

    page_relative = relative_posix(workspace, page_path)
    page_title = title if title is not None else _derive_title(normalized_page_id)
    claim_block_id = _derive_marker_id(
        normalized_page_id, prefix="claim_block."
    )
    commentary_id = _derive_marker_id(normalized_page_id, prefix="commentary.")
    stub_text = _render_stub(
        normalized_page_id=normalized_page_id,
        page_class=resolved_class,
        title=page_title,
        claim_block_id=claim_block_id,
        commentary_id=commentary_id,
        framework_profile=framework_profile,
    )
    try:
        parse_page(stub_text)
    except PageParseError as exc:  # pragma: no cover - defensive
        raise PageCreateError(
            f"internal error: generated stub failed parse_page validation: {exc}"
        ) from exc

    with operation(
        workspace,
        op_kind="page_create",
        planned_writes=[page_relative],
    ) as ctx:
        if os.path.lexists(page_path):
            return PageCreateResult(
                page_id=normalized_page_id,
                page_class=resolved_class,
                page_path=page_relative,
                claim_block_id=claim_block_id,
                commentary_id=commentary_id,
                status="draft",
                op_id=ctx.op_id,
                refusal_reason=(
                    f"page already exists at {page_relative!r}; "
                    "remove it before re-creating"
                ),
            )

        collision_reason = _publish_stub(
            page_path,
            stub_text,
            framework_profile=framework_profile,
            page_relative=page_relative,
            class_root=(workspace.pages / _PAGE_CLASSES[resolved_class]).resolve(),
        )
        if collision_reason is not None:
            return PageCreateResult(
                page_id=normalized_page_id,
                page_class=resolved_class,
                page_path=page_relative,
                claim_block_id=claim_block_id,
                commentary_id=commentary_id,
                status="draft",
                op_id=ctx.op_id,
                refusal_reason=collision_reason,
            )

        ctx.entry.touched_files.append(page_relative)
        ctx.entry.notes.append(
            f"created page stub {normalized_page_id!r} -> {page_relative}"
        )

        return PageCreateResult(
            page_id=normalized_page_id,
            page_class=resolved_class,
            page_path=page_relative,
            claim_block_id=claim_block_id,
            commentary_id=commentary_id,
            status="draft",
            op_id=ctx.op_id,
        )


def _validate_framework_profile(framework_profile: str | None) -> None:
    """Refuse any opt-in value other than the single accepted candidate.

    Runs before the operation context opens and before any directory
    or file is created, so an unsupported value leaves the workspace
    untouched.
    """
    if framework_profile is None:
        return
    if framework_profile != CANDIDATE_FRAMEWORK_PROFILE:
        raise PageCreateError(
            f"unsupported framework_profile {framework_profile!r}; the only "
            f"accepted value is {CANDIDATE_FRAMEWORK_PROFILE!r}"
        )


def _create_parents(target_dir: Path, class_root: Path) -> list[Path]:
    """Create missing output parents, returning only those this call created.

    Creation stops at ``class_root``, which is the pre-existing page-class
    directory and is never tracked or removed.
    """
    class_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    current = class_root
    for part in target_dir.relative_to(class_root).parts:
        current = current / part
        if not current.exists():
            current.mkdir()
            created.append(current)
    return created


def _remove_created_dirs(created: list[Path]) -> None:
    """Remove operation-created directories bottom-up, only while empty.

    A directory that concurrent work has filled is preserved, and the
    walk stops there. Nothing is ever removed recursively, and another
    writer's content is never deleted.
    """
    for directory in reversed(created):
        try:
            directory.rmdir()
        except OSError:
            break


def _write_temp(fd: int, stub_text: str, *, framework_profile: str | None) -> None:
    """Write the stub through the owned descriptor.

    Handle acquisition, write, flush, and close are distinct seams. A
    failure at any of them closes the descriptor and propagates the
    original error unchanged.
    """
    try:
        if framework_profile is None:
            # Accepted legacy writer semantics: text mode, platform newlines.
            handle = os.fdopen(fd, "w", encoding="utf-8", newline=None)
            payload: str | bytes = stub_text
        else:
            # Deterministic LF bytes, independent of host platform.
            handle = os.fdopen(fd, "wb")
            payload = stub_text.encode("utf-8")
    except BaseException:
        os.close(fd)
        raise

    try:
        handle.write(payload)
        handle.flush()
    except BaseException:
        try:
            handle.close()
        except OSError:
            pass
        raise
    handle.close()


def _abandon(
    tmp: Path,
    created_dirs: list[Path],
    *,
    page_relative: str,
    primary: BaseException | None,
) -> None:
    """Discard this operation's own temp, then its own empty directories.

    A cleanup denial is never swallowed. It is surfaced as a bounded,
    deterministic error that keeps the primary failure context and names
    neither a machine-local path nor a traceback.
    """
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        detail = (
            "this operation's temporary artifact could not be removed and is "
            f"retained beside {page_relative!r} for reconciliation"
        )
        if primary is not None:
            raise PageCreateError(f"{primary}; additionally {detail}") from primary
        raise PageCreateError(detail) from exc
    _remove_created_dirs(created_dirs)


def _publish_stub(
    page_path: Path,
    stub_text: str,
    *,
    framework_profile: str | None,
    page_relative: str,
    class_root: Path,
) -> str | None:
    """Write an owned temp, validate, then publish with atomic no-clobber.

    Publication uses :func:`os.link`, a same-directory hard link that the
    operating system refuses when the final name already exists. That
    refusal is the no-clobber guarantee itself, not a check performed
    before an overwrite-capable transfer: no ``os.replace``, no
    ``Path.replace``, and no check-then-transfer pair is used. A
    competitor that appears at any point before the link therefore wins
    unchanged, and this function returns a refusal reason instead of
    raising, so the operation completes and releases its lock normally.

    Returns ``None`` on success or the refusal reason on collision.
    """
    created_dirs = _create_parents(page_path.parent, class_root)
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(page_path.parent), prefix=page_path.name + ".", suffix=".tmp"
        )
    except BaseException:
        _remove_created_dirs(created_dirs)
        raise

    tmp = Path(tmp_name)
    collision = False
    try:
        _write_temp(fd, stub_text, framework_profile=framework_profile)
        if framework_profile is not None:
            _require_exact_candidate(tmp, page_relative=page_relative)
        try:
            os.link(tmp, page_path)
        except FileExistsError:
            collision = True
        except OSError as exc:
            raise PageCreateError(
                f"cannot publish {page_relative!r}: the target filesystem did not "
                "provide the required atomic no-clobber link primitive"
            ) from exc
    except BaseException as primary:
        _abandon(tmp, created_dirs, page_relative=page_relative, primary=primary)
        raise

    if collision:
        _abandon(tmp, created_dirs, page_relative=page_relative, primary=None)
        return (
            f"page already exists at {page_relative!r}; "
            "remove it before re-creating"
        )

    # Published. Only this operation's temp name remains to be dropped.
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise PageCreateError(
            f"published {page_relative!r} but could not remove this operation's "
            "temporary name beside it; the page is complete and the retained "
            "artifact is left for reconciliation"
        ) from exc
    return None


def _require_exact_candidate(tmp: Path, *, page_relative: str) -> None:
    """Validate the exact candidate bytes through the accepted M002 surface.

    Only the exact R14 outcome is accepted. Any other observation is an
    internal creation failure, never permission to reinterpret M002.
    """
    observation = observe_page_okf(tmp)
    expected = (
        ("read_status", "ok"),
        ("okf_concept_result", "pass"),
        ("okf_concept_reason", None),
        ("framework_profile_result", "pass"),
        ("framework_profile_reason", None),
        ("declared_framework_profile", CANDIDATE_FRAMEWORK_PROFILE),
        ("execution_eligibility", "not_evaluated"),
    )
    mismatches = [
        f"{field}={getattr(observation, field)!r} (expected {value!r})"
        for field, value in expected
        if getattr(observation, field) != value
    ]
    if mismatches:
        raise PageCreateError(
            "internal error: generated candidate page for "
            f"{page_relative!r} did not observe as the exact accepted "
            "profile outcome: " + "; ".join(mismatches)
        )


def _validate_page_id(page_id: str) -> tuple[str, str | None]:
    """Validate ``page_id`` and return ``(normalized, inferred_class)``.

    The normalized form is the page_id as written; only refusal-worthy
    shapes are rejected. The inferred class is set when the first slash
    segment unambiguously names a recognized class (or its plural
    directory form); otherwise None.
    """
    if not isinstance(page_id, str) or not page_id:
        raise PageCreateError("page_id must be a non-empty string")
    if page_id.endswith(".md"):
        raise PageCreateError(
            f"page_id must not end with .md (got {page_id!r}); "
            "pass the bare id, e.g. 'concept/foo'"
        )
    if "\\" in page_id:
        raise PageCreateError(
            f"page_id must use forward slashes only (got {page_id!r})"
        )
    if page_id.startswith("/"):
        raise PageCreateError(
            f"page_id must be workspace-relative (got {page_id!r}); "
            "do not start with '/'"
        )
    if Path(page_id).is_absolute():
        raise PageCreateError(
            f"page_id must be workspace-relative (got {page_id!r})"
        )

    segments = page_id.split("/")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise PageCreateError(
                f"page_id has empty or traversal segment in {page_id!r}; "
                "'.' / '..' / empty segments are not allowed"
            )
        if not _PAGE_ID_SEGMENT_RE.match(segment):
            raise PageCreateError(
                f"page_id segment {segment!r} in {page_id!r} contains "
                "invalid characters; only letters, digits, '.', '_', "
                "and '-' are accepted"
            )

    inferred = _CLASS_PREFIXES.get(segments[0]) if len(segments) >= 2 else None
    return page_id, inferred


def _resolve_page_class(
    *,
    page_class: str | None,
    inferred_class: str | None,
) -> str:
    """Pick a single canonical page class from the explicit flag + inferred prefix."""
    if page_class is not None:
        if page_class not in _PAGE_CLASSES:
            raise PageCreateError(
                f"unknown --page-class {page_class!r}; accepted: "
                f"{sorted(_PAGE_CLASSES)}"
            )
        if inferred_class is not None and inferred_class != page_class:
            raise PageCreateError(
                f"conflicting page class: --page-class={page_class!r} but "
                f"page_id prefix infers {inferred_class!r}; pass a matching "
                "--page-class or drop the prefix from page_id"
            )
        return page_class
    if inferred_class is None:
        raise PageCreateError(
            "cannot infer page class from page_id; pass --page-class "
            f"(one of {sorted(_PAGE_CLASSES)})"
        )
    return inferred_class


def _resolve_page_path(
    workspace: Workspace,
    *,
    normalized_page_id: str,
    page_class: str,
) -> Path:
    """Map ``(page_id, page_class)`` to the on-disk page path.

    Strips a recognized class prefix from the page_id when present so
    `concept/foo` lands at `pages/concepts/foo.md` rather than
    `pages/concepts/concept/foo.md`. Raises :class:`PageCreateError`
    if the resolved path would escape the selected class directory.
    """
    class_dir_name = _PAGE_CLASSES[page_class]
    class_root = (workspace.pages / class_dir_name).resolve()

    segments = normalized_page_id.split("/")
    if len(segments) >= 2 and _CLASS_PREFIXES.get(segments[0]) == page_class:
        tail_segments = segments[1:]
    else:
        tail_segments = segments

    if not tail_segments or any(s in ("", ".", "..") for s in tail_segments):
        raise PageCreateError(
            f"page_id {normalized_page_id!r} resolved to an empty or "
            "traversal-only path tail under the selected class directory"
        )

    # Append ``.md`` to the final tail segment directly. Do NOT use
    # ``Path.with_suffix(".md")`` here — for a dotted page id like
    # ``concept/foo.bar`` the joined path would be
    # ``pages/concepts/foo.bar`` and ``with_suffix(".md")`` would
    # replace the existing ``.bar`` suffix, yielding
    # ``pages/concepts/foo.md`` and silently dropping the rest of
    # the id. Renaming the last segment with ``+ ".md"`` preserves
    # the full tail.
    suffixed_tail = list(tail_segments)
    suffixed_tail[-1] = suffixed_tail[-1] + ".md"
    candidate = class_root.joinpath(*suffixed_tail)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(class_root)
    except ValueError as exc:
        raise PageCreateError(
            f"page_id {normalized_page_id!r} would escape the "
            f"{class_dir_name}/ directory"
        ) from exc
    return resolved


def _derive_title(normalized_page_id: str) -> str:
    """Derive a human-readable title from the final page_id segment."""
    tail = normalized_page_id.split("/")[-1]
    parts = re.split(r"[-_.]+", tail)
    cleaned = [p for p in parts if p]
    if not cleaned:
        return tail
    return " ".join(word.capitalize() for word in cleaned)


def _derive_marker_id(normalized_page_id: str, *, prefix: str) -> str:
    """Deterministic marker id: replace ``/`` with ``.`` and prefix."""
    body = normalized_page_id.replace("/", ".")
    marker = f"{prefix}{body}"
    if not _MARKER_ID_RE.match(marker):  # pragma: no cover - defensive
        raise PageCreateError(
            f"derived marker id {marker!r} does not match the parser "
            "regex; check the page_id for unsupported characters"
        )
    return marker


def _render_stub(
    *,
    normalized_page_id: str,
    page_class: str,
    title: str,
    claim_block_id: str,
    commentary_id: str,
    framework_profile: str | None = None,
) -> str:
    """Build the variant-(B) page stub text.

    With the opt-in, exactly two frontmatter fields are prepended. They
    form one removable compatibility block: deleting exactly
    ``type: page\\nframework_profile: "0.1-rc.1"\\n`` recovers the
    canonical LF legacy rendering this function returns without it.
    """
    profile_block = (
        ""
        if framework_profile is None
        else ("type: page\n" f'framework_profile: "{framework_profile}"\n')
    )
    return (
        "---\n"
        f"{profile_block}"
        f"page_id: {normalized_page_id}\n"
        f"page_class: {page_class}\n"
        "write_policy: mixed\n"
        "status: draft\n"
        "---\n"
        "\n"
        f"# {title}\n"
        "\n"
        f"<!-- llloom:claim-block id={claim_block_id} -->\n"
        "\n"
        f"<!-- /llloom:claim-block -->\n"
        "\n"
        f"<!-- llloom:commentary id={commentary_id} owner=human -->\n"
        "\n"
        f"<!-- /llloom:commentary -->\n"
    )
