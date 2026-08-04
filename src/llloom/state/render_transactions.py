"""Render transaction staging (Slice 074).

Stages page bytes and the full render-fingerprint snapshot under
``state/transactions/<op_id>/`` before committing them to final
workspace paths. The commit replaces every final path via
``tmp.replace(target)`` and then removes the transaction directory.
An interruption before the commit phase leaves final paths
byte-identical to before the operation; an interruption during
commit leaves a diagnosable transaction directory on disk and the
existing operation-level journal/lock semantics carry the
recoverable-state signal that ``reconcile`` already uses.

Transaction directories are **in-flight write buffers**, not
sidecars. They are authoritative work-in-progress, not derived
from canonical YAML, and not rebuildable once an operation is
abandoned. The success path removes them. ``reconcile`` cleans an
abandoned transaction directory only when it is also clearing the
corresponding stale lock — the predicate that already governs
journal-backed recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from llloom.ops._context import iso_now, relative_posix
from llloom.workspace.layout import Workspace


_MANIFEST_FILENAME = "manifest.yaml"
_PAGES_DIRNAME = "pages"
_STAGED_FINGERPRINTS_FILENAME = "render_fingerprints.yaml"


class RenderTransactionError(Exception):
    """Raised when a render transaction cannot stage or commit."""


@dataclass(frozen=True)
class _StagedPage:
    page_id: str
    final_path: Path
    staged_path: Path
    wrote: bool


class RenderTransaction:
    """Stage-then-commit primitive for one render operation.

    Usage from ``ops/render.py``::

        txn = RenderTransaction(workspace, op_id)
        for plan in items:
            txn.stage_page(page_id, page_path, original, new_text)
        txn.stage_fingerprints(planned_fingerprints)
        txn.commit()

    On success ``commit`` removes the transaction directory. On any
    raised exception during staging or commit, the directory is
    left in place with a manifest whose ``status`` field tells the
    operator (or ``reconcile``) what happened. The class never
    catches exceptions itself; the operation context manager
    surrounding the call decides what to do with them.
    """

    def __init__(self, workspace: Workspace, op_id: str) -> None:
        self._workspace = workspace
        self._op_id = op_id
        self._dir: Path = workspace.state_transactions / op_id
        self._pages_dir: Path = self._dir / _PAGES_DIRNAME
        self._staged_pages: list[_StagedPage] = []
        self._planned_fingerprints: dict[str, str] | None = None
        self._committed = False

    @property
    def root(self) -> Path:
        return self._dir

    @property
    def op_id(self) -> str:
        return self._op_id

    @property
    def manifest_path(self) -> Path:
        return self._dir / _MANIFEST_FILENAME

    @property
    def staged_fingerprints_path(self) -> Path:
        return self._dir / _STAGED_FINGERPRINTS_FILENAME

    def stage_page(
        self,
        *,
        page_id: str,
        final_path: Path,
        original_text: str,
        new_text: str,
    ) -> bool:
        """Write the planned page bytes into the transaction directory.

        Returns True when the bytes differ from the page on disk
        (the page is a real candidate for commit), False otherwise.
        The staged file is written regardless of the diff so the
        commit phase can verify the staged content matches what
        was planned.
        """
        self._pages_dir.mkdir(parents=True, exist_ok=True)
        staged = self._pages_dir / _staged_page_relative(self._workspace, final_path)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(new_text, encoding="utf-8")
        wrote = new_text != original_text
        self._staged_pages.append(
            _StagedPage(
                page_id=page_id,
                final_path=final_path,
                staged_path=staged,
                wrote=wrote,
            )
        )
        return wrote

    def stage_fingerprints(self, planned_fingerprints: dict[str, str]) -> None:
        """Write the planned full fingerprint snapshot into the txn.

        The snapshot is the complete ``{page_id: sha256}`` mapping
        that will replace the current ``state/render_fingerprints.yaml``
        on commit. Callers are expected to seed it from
        ``FingerprintStore.load()`` and overlay the new values for
        the pages this transaction renders.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        self._planned_fingerprints = dict(planned_fingerprints)
        payload = {"fingerprints": dict(planned_fingerprints)}
        self.staged_fingerprints_path.write_text(
            yaml.safe_dump(payload, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )

    def write_manifest(self, *, status: str, notes: list[str] | None = None) -> None:
        """Persist a diagnostic manifest describing the txn state.

        ``status`` is one of ``staged`` / ``committing`` /
        ``committed`` / ``aborted``. The manifest names every staged
        page (with its planned final path) and the staged
        fingerprint file so an operator or ``reconcile`` can read
        the transaction directory and understand what was in flight.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "op_id": self._op_id,
            "op_kind": "render",
            "status": status,
            "updated_at": iso_now(),
            "pages": [
                {
                    "page_id": p.page_id,
                    "final_path": relative_posix(self._workspace, p.final_path),
                    "staged_path": relative_posix(self._workspace, p.staged_path),
                    "wrote": p.wrote,
                }
                for p in self._staged_pages
            ],
            "fingerprints": {
                "final_path": relative_posix(
                    self._workspace, self._workspace.render_fingerprints
                ),
                "staged_path": relative_posix(
                    self._workspace, self.staged_fingerprints_path
                ),
            },
            "notes": list(notes or []),
        }
        tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        tmp.replace(self.manifest_path)

    def commit(self) -> tuple[list[str], list[str]]:
        """Atomically replace every final path with the staged file.

        Page commits run first (preserving Slice 072's "stale
        fingerprint can be recovered by lint/reconcile if commit
        is interrupted between pages and fingerprints" property);
        the fingerprint file commits last. On success the
        transaction directory is removed and the function returns
        ``(rendered_relpaths, unchanged_relpaths)``. On failure the
        caller is expected to let the exception propagate; the
        operation context manager leaves the journal in_progress
        and the lock held for ``reconcile`` to triage.
        """
        if self._planned_fingerprints is None:
            raise RenderTransactionError(
                f"render transaction {self._op_id} has no staged fingerprints"
            )
        self.write_manifest(status="committing")

        rendered: list[str] = []
        unchanged: list[str] = []
        for staged in self._staged_pages:
            rel = relative_posix(self._workspace, staged.final_path)
            if not staged.wrote:
                unchanged.append(rel)
                continue
            staged.final_path.parent.mkdir(parents=True, exist_ok=True)
            # tmp.replace(target) directly — never unlink target first.
            staged.staged_path.replace(staged.final_path)
            rendered.append(rel)

        # Fingerprint commit last so a between-pages interruption
        # leaves stale fingerprints that lint / reconcile already
        # know how to recover.
        target = self._workspace.render_fingerprints
        target.parent.mkdir(parents=True, exist_ok=True)
        self.staged_fingerprints_path.replace(target)

        self.write_manifest(
            status="committed",
            notes=[
                f"committed {len(rendered)} rendered + "
                f"{len(unchanged)} unchanged pages"
            ],
        )
        # Success: remove the transaction directory.
        _remove_dir_tree(self._dir)
        self._committed = True
        return rendered, unchanged


def _staged_page_relative(workspace: Workspace, final_path: Path) -> Path:
    """Map a workspace-final page path to its staged-side counterpart.

    ``state/transactions/<op_id>/pages/<workspace-relative-page-path>``.
    """
    rel = final_path.resolve().relative_to(workspace.root.resolve())
    return Path(*rel.parts)


def _remove_dir_tree(path: Path) -> None:
    """Best-effort recursive removal of a transaction directory.

    Uses ``shutil.rmtree`` indirectly to keep the dependency surface
    in the stdlib. Missing-path is not an error (commit may have
    already run on a retry).
    """
    import shutil

    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
