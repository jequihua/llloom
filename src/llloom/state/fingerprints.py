"""Render fingerprints store.

Thin YAML sidecar at ``state/render_fingerprints.yaml`` mapping
``page_id -> fingerprint`` so ``reconcile`` can detect stale rendered
pages deterministically.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from llloom.workspace.layout import Workspace


class FingerprintStore:
    """File-backed render fingerprints."""

    def __init__(self, workspace: Workspace) -> None:
        self._path: Path = workspace.render_fingerprints

    def load(self) -> dict[str, str]:
        if not self._path.is_file():
            return {}
        data = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{self._path} must be a YAML mapping")
        fingerprints = data.get("fingerprints", {}) or {}
        if not isinstance(fingerprints, dict):
            raise ValueError(f"{self._path}: fingerprints must be a mapping")
        return {str(k): str(v) for k, v in fingerprints.items()}

    def save(self, fingerprints: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprints": dict(fingerprints)}
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(payload, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def set(self, page_id: str, fingerprint: str) -> None:
        fps = self.load()
        fps[page_id] = fingerprint
        self.save(fps)

    def get(self, page_id: str) -> str | None:
        return self.load().get(page_id)

