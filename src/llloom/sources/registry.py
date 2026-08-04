"""Source registry: register, hash, classify, and track sources.

Stores one YAML record per source under
``state/source_registry/<source_id>.yaml``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import yaml

from llloom.workspace.layout import Workspace


SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class SourceRegistryError(Exception):
    """Raised for registry-level failures (duplicate ids, modified evidence, etc.)."""


@dataclass
class SourceRecord:
    """Authoritative source-registry record.

    ``raw_path`` is repo-relative (POSIX style). ``content_hash`` is the
    SHA-256 of the raw bytes, prefixed with ``sha256:``.
    """

    source_id: str
    source_class: str
    raw_path: str
    content_hash: str
    byte_size: int
    status: str = "registered"  # registered | retracted
    registered_at: str = ""
    last_seen_at: str = ""
    retracted_at: str | None = None
    retraction_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict:
        data = asdict(self)
        # Drop None fields so YAML stays tidy.
        return {k: v for k, v in data.items() if v is not None}


class SourceRegistry:
    """File-backed registry under ``state/source_registry/``."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._dir = workspace.state_source_registry

    # ---- id helpers -----------------------------------------------------

    @staticmethod
    def validate_source_id(source_id: str) -> None:
        if not SOURCE_ID_PATTERN.match(source_id):
            raise SourceRegistryError(
                f"invalid source_id {source_id!r}: must match {SOURCE_ID_PATTERN.pattern}"
            )

    @staticmethod
    def derive_source_id(raw_path: Path) -> str:
        """Derive a stable default source_id from a raw path.

        Lower-cases the stem, replaces non-id characters with dots, and
        collapses repeats. Callers may still override the id explicitly.
        """
        stem = raw_path.stem.lower()
        slug = re.sub(r"[^a-z0-9._-]+", ".", stem).strip(".")
        slug = re.sub(r"\.{2,}", ".", slug) or "source"
        candidate = f"src.{slug}"
        # Enforce length.
        return candidate[:128]

    # ---- persistence ----------------------------------------------------

    def record_path(self, source_id: str) -> Path:
        return self._dir / f"{source_id}.yaml"

    def list_ids(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        return sorted(p.stem for p in self._dir.glob("*.yaml"))

    def iter_records(self) -> Iterator[SourceRecord]:
        for sid in self.list_ids():
            yield self.load(sid)

    def exists(self, source_id: str) -> bool:
        return self.record_path(source_id).is_file()

    def load(self, source_id: str) -> SourceRecord:
        path = self.record_path(source_id)
        if not path.is_file():
            raise SourceRegistryError(f"source not registered: {source_id}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return SourceRecord(
            source_id=str(data["source_id"]),
            source_class=str(data["source_class"]),
            raw_path=str(data["raw_path"]),
            content_hash=str(data["content_hash"]),
            byte_size=int(data["byte_size"]),
            status=str(data.get("status", "registered")),
            registered_at=str(data.get("registered_at", "")),
            last_seen_at=str(data.get("last_seen_at", "")),
            retracted_at=data.get("retracted_at"),
            retraction_reason=data.get("retraction_reason"),
            notes=list(data.get("notes", []) or []),
        )

    def save(self, record: SourceRecord) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self.record_path(record.source_id)
        payload = record.to_mapping()
        _atomic_write_yaml(path, payload)

    # ---- hashing --------------------------------------------------------

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    @staticmethod
    def hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()

    # ---- registration ---------------------------------------------------

    def register(
        self,
        source_id: str,
        raw_path: Path,
        source_class: str,
    ) -> tuple[SourceRecord, str]:
        """Register (or re-observe) ``raw_path`` under ``source_id``.

        Returns (record, state) where state is one of:

        - ``"new"``: first-time registration
        - ``"unchanged"``: previously registered, hash unchanged
        - ``"modified"``: previously registered, hash changed (refused)

        ``"modified"`` raises SourceRegistryError because raw evidence is
        immutable.
        """
        self.validate_source_id(source_id)
        if not raw_path.is_file():
            raise SourceRegistryError(f"raw source file not found: {raw_path}")

        new_hash = self.hash_file(raw_path)
        rel = _relative_posix(raw_path, self._workspace.root)
        size = raw_path.stat().st_size
        now = _now_iso()

        if self.exists(source_id):
            existing = self.load(source_id)
            if existing.content_hash != new_hash:
                raise SourceRegistryError(
                    f"source {source_id}: raw evidence changed "
                    f"(recorded hash {existing.content_hash}, observed {new_hash}); "
                    f"raw evidence is immutable"
                )
            if existing.source_class != source_class:
                raise SourceRegistryError(
                    f"source {source_id}: class mismatch "
                    f"(recorded {existing.source_class}, requested {source_class})"
                )
            existing.last_seen_at = now
            self.save(existing)
            return existing, "unchanged"

        record = SourceRecord(
            source_id=source_id,
            source_class=source_class,
            raw_path=rel,
            content_hash=new_hash,
            byte_size=size,
            status="registered",
            registered_at=now,
            last_seen_at=now,
        )
        self.save(record)
        return record, "new"

    def mark_retracted(self, source_id: str, reason: str | None = None) -> SourceRecord:
        record = self.load(source_id)
        now = _now_iso()
        record.status = "retracted"
        record.retracted_at = now
        record.retraction_reason = reason
        self.save(record)
        return record

    def raw_text(self, record: SourceRecord) -> str:
        """Read raw source text as UTF-8."""
        return (self._workspace.root / record.raw_path).read_text(encoding="utf-8")


# ---- helpers -----------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _relative_posix(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SourceRegistryError(
            f"raw path {path} is outside workspace root {root}"
        ) from exc
    return rel.as_posix()


def _atomic_write_yaml(path: Path, data: dict) -> None:
    """Write YAML via temp-file-and-rename so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = yaml.safe_dump(data, sort_keys=True, allow_unicode=True)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

