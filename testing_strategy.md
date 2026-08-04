# Testing llloom

Contributor guide for running the product test suite. For normative behavior,
see the root contracts: `architecture_contract.md`, `public_api_contract.md`,
and `package_scope.md`.

## Setup

From the repository root, install the contributor dependencies:

```bash
python -m pip install -e ".[dev]"
```

Requires Python 3.11+.

## Suite Layout

Tests live under `tests/` in three release-contained layers:

- `tests/unit/` — focused unit tests for individual modules and data
  structures.
- `tests/contract/` — public-surface contracts: CLI/JSON shapes, public API,
  workspace layout, lock and journal behavior, and packaging-facing
  guarantees.
- `tests/integration/` — cross-module flows: ingest → verify → render →
  query cycles, fixture parity with the pinned interoperability corpus, and
  end-to-end workspace scenarios.

## Running The Suite

The canonical command runs the three layers from the repository root:

```bash
python -m pytest tests/contract tests/unit tests/integration
```

On Windows, the accepted partition keeps the lock-owner process-probe module
isolated, because its console-PID probe can interrupt a monolithic capture:

```bash
python -m pytest tests/contract tests/unit tests/integration --ignore=tests/contract/test_lock_owner_metadata.py
python -m pytest tests/contract/test_lock_owner_metadata.py
```

## Expected Skips

Some tests skip by design rather than failing:

- optional dependency smokes (for example the tree-sitter language smokes)
  skip until the matching extra is installed;
- symlink-construction cases skip on platforms without the required
  privilege (Windows without symlink permission);
- a small number of explicit guards skip assets that are intentionally not
  part of the release-contained tree.

A skip in these classes is an accepted outcome, not a failure. If you install
the optional extras or run on a platform with symlink privilege, those rows
execute normally.
