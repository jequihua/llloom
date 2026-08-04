"""Shared test fixtures.

Every test that needs a workspace should request ``fresh_workspace``.
It builds a fresh repo-native workspace in a tmp path and copies a
subset of the real fixture corpus into ``raw/sources/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from llloom.workspace.layout import Workspace


def _find_package_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src" / "llloom").is_dir()
        ):
            return candidate
    raise RuntimeError(f"could not locate llloom package root from {start}")


def _fixture_corpus_root(package_root: Path) -> Path:
    in_package = package_root / "docs" / "fixture_corpus"
    if in_package.is_dir():
        return in_package
    return package_root.parent / "docs" / "fixture_corpus"


REPO_ROOT = _find_package_root(Path(__file__).resolve())
FIXTURE_CORPUS = _fixture_corpus_root(REPO_ROOT)
SYNTHETIC_ROOT = Path(__file__).resolve().parent / "fixtures" / "synthetic"


@pytest.fixture
def fresh_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.init(tmp_path)
    return ws


@pytest.fixture
def fixture_corpus_root() -> Path:
    return FIXTURE_CORPUS


@pytest.fixture
def synthetic_root() -> Path:
    return SYNTHETIC_ROOT


def copy_fixture_to_raw(workspace: Workspace, src: Path) -> Path:
    """Copy ``src`` into the workspace ``raw/sources/`` and return new path."""
    dst = workspace.raw_sources / src.name
    shutil.copyfile(src, dst)
    return dst

