"""Unit tests for the source registry and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from llloom.sources.registry import SourceRegistry, SourceRegistryError
from llloom.workspace.layout import Workspace


def _write_raw(ws: Workspace, name: str, body: str) -> Path:
    path = ws.raw_sources / name
    path.write_text(body, encoding="utf-8")
    return path


def test_hash_bytes_is_sha256(tmp_path: Path) -> None:
    assert SourceRegistry.hash_bytes(b"hello").startswith("sha256:")


def test_validate_and_derive_ids(tmp_path: Path) -> None:
    SourceRegistry.validate_source_id("src.example")
    with pytest.raises(SourceRegistryError):
        SourceRegistry.validate_source_id("Bad Id")
    derived = SourceRegistry.derive_source_id(Path("Hello World.md"))
    assert derived.startswith("src.") and " " not in derived


def test_register_new_then_unchanged(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    registry = SourceRegistry(ws)
    path = _write_raw(ws, "alpha.md", "# Alpha\n\nBody.\n")
    record, state = registry.register(
        source_id="src.alpha", raw_path=path, source_class="markdown_prose"
    )
    assert state == "new"
    assert record.content_hash.startswith("sha256:")
    # Re-register unchanged.
    _, state2 = registry.register(
        source_id="src.alpha", raw_path=path, source_class="markdown_prose"
    )
    assert state2 == "unchanged"


def test_register_modified_refused(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    registry = SourceRegistry(ws)
    path = _write_raw(ws, "beta.md", "# Beta\n")
    registry.register(
        source_id="src.beta", raw_path=path, source_class="markdown_prose"
    )
    path.write_text("# Beta\n\nAltered evidence.\n", encoding="utf-8")
    with pytest.raises(SourceRegistryError) as exc:
        registry.register(
            source_id="src.beta", raw_path=path, source_class="markdown_prose"
        )
    assert "immutable" in str(exc.value)


def test_mark_retracted(tmp_path: Path) -> None:
    ws = Workspace.init(tmp_path)
    registry = SourceRegistry(ws)
    path = _write_raw(ws, "gamma.md", "# Gamma\n")
    registry.register(
        source_id="src.gamma", raw_path=path, source_class="markdown_prose"
    )
    record = registry.mark_retracted("src.gamma", reason="test")
    assert record.status == "retracted"
    assert record.retraction_reason == "test"

