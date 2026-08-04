"""Unit tests for workspace layout validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.workspace.layout import REQUIRED_DIRS, REQUIRED_SCHEMA_FILES, Workspace, WorkspaceError


def test_init_creates_required_layout(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    assert ws.root == tmp_path.resolve()
    for rel in REQUIRED_DIRS:
        assert (tmp_path / rel).is_dir(), f"missing {rel}"
    for rel in REQUIRED_SCHEMA_FILES:
        assert (tmp_path / rel).is_file(), f"missing {rel}"
    assert (tmp_path / "pages" / "overview.md").is_file()
    assert (tmp_path / "state" / "render_fingerprints.yaml").is_file()


def test_load_validates_layout(tmp_path: Path) -> None:
    Workspace.init(tmp_path)
    # Remove one required directory and expect failure.
    (tmp_path / "claims" / "entities").rmdir()
    with pytest.raises(WorkspaceError) as exc:
        Workspace.load(tmp_path)
    assert "claims/entities" in str(exc.value)


def test_named_paths(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    assert ws.raw_sources == tmp_path / "raw" / "sources"
    assert ws.claims_entities == tmp_path / "claims" / "entities"
    assert ws.claims_merge_proposals == tmp_path / "claims" / "merge_proposals"
    assert ws.state_locks == tmp_path / "state" / "locks"
    assert ws.state_journals == tmp_path / "state" / "journals"
    assert ws.render_fingerprints == tmp_path / "state" / "render_fingerprints.yaml"

